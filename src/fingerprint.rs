use anyhow::{Result, anyhow, bail, ensure};

const SHA1_INITIAL_STATE: [u32; 5] = [
    0x6745_2301,
    0xEFCD_AB89,
    0x98BA_DCFE,
    0x1032_5476,
    0xC3D2_E1F0,
];

const ED25519_OID: [u8; 9] = [0x2B, 0x06, 0x01, 0x04, 0x01, 0xDA, 0x47, 0x0F, 0x01];

#[derive(Clone, Copy, Debug)]
pub struct HexPrefix {
    nibbles: usize,
    aligned_value: u64,
    mask: u64,
}

impl HexPrefix {
    pub fn parse(prefix: &str) -> Result<Self> {
        let trimmed = prefix.trim();
        let hex = trimmed
            .strip_prefix("0x")
            .or_else(|| trimmed.strip_prefix("0X"))
            .unwrap_or(trimmed);

        ensure!(!hex.is_empty(), "prefix must not be empty");
        ensure!(
            hex.len() <= 16,
            "key ID prefixes can be at most 16 hex characters"
        );

        let mut value = 0u64;
        for byte in hex.bytes() {
            let nibble = decode_nibble(byte).ok_or_else(|| {
                anyhow!("prefix contains a non-hex character: {:?}", byte as char)
            })?;
            value = (value << 4) | u64::from(nibble);
        }

        let used_bits = hex.len() * 4;
        let shift = 64usize.saturating_sub(used_bits);
        let mask = if used_bits == 64 {
            u64::MAX
        } else {
            u64::MAX << shift
        };

        Ok(Self {
            nibbles: hex.len(),
            aligned_value: value << shift,
            mask,
        })
    }

    pub fn matches(&self, key_id: u64) -> bool {
        (key_id & self.mask) == self.aligned_value
    }

    /// Number of hex digits within the prefix region that differ from `key_id`.
    pub fn error_count(&self, key_id: u64) -> u32 {
        let diff = (key_id ^ self.aligned_value) & self.mask;
        // Collapse each 4-bit nibble to its low bit (set iff the nibble is
        // nonzero), then count the set bits — that's the number of wrong digits.
        let nonzero_nibbles =
            (diff | (diff >> 1) | (diff >> 2) | (diff >> 3)) & 0x1111_1111_1111_1111u64;
        nonzero_nibbles.count_ones()
    }

    pub fn matches_with_error(&self, key_id: u64, max_error: u32) -> bool {
        self.error_count(key_id) <= max_error
    }

    pub fn normalized(&self) -> String {
        let raw = format!("{:016X}", self.aligned_value);
        raw[..self.nibbles].to_string()
    }

    pub fn search_space_size(&self) -> u128 {
        1u128 << (self.nibbles * 4)
    }

    /// Number of distinct key IDs whose prefix region is within `max_error` hex
    /// digits of this prefix: sum_{k=0}^{e} C(n, k) * 15^k, where n is the
    /// number of nibbles and e = min(max_error, n).
    pub fn match_count(&self, max_error: u32) -> u128 {
        let n = self.nibbles as u128;
        let e = u128::from(max_error).min(n);
        let mut count = 0u128;
        let mut binom = 1u128; // C(n, 0)
        let mut pow15 = 1u128; // 15^0
        let mut k = 0u128;
        loop {
            count += binom * pow15;
            if k == e {
                break;
            }
            // Advance to C(n, k+1) and 15^(k+1).
            binom = binom * (n - k) / (k + 1);
            pow15 *= 15;
            k += 1;
        }
        count
    }

    pub(crate) fn mask_words(&self) -> (u32, u32) {
        ((self.mask >> 32) as u32, self.mask as u32)
    }

    pub(crate) fn value_words(&self) -> (u32, u32) {
        ((self.aligned_value >> 32) as u32, self.aligned_value as u32)
    }
}

pub const MAX_PREFIXES: usize = 64;

#[derive(Clone, Debug)]
pub struct HexPrefixSet {
    prefixes: Vec<HexPrefix>,
    max_error: u32,
}

impl HexPrefixSet {
    pub fn parse<I, S>(items: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let prefixes = items
            .into_iter()
            .map(|item| HexPrefix::parse(item.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        Self::from_prefixes(prefixes)
    }

    pub fn from_prefixes(prefixes: Vec<HexPrefix>) -> Result<Self> {
        ensure!(!prefixes.is_empty(), "at least one prefix is required");
        ensure!(
            prefixes.len() <= MAX_PREFIXES,
            "at most {MAX_PREFIXES} prefixes are supported, got {}",
            prefixes.len()
        );
        Ok(Self {
            prefixes,
            max_error: 0,
        })
    }

    pub fn with_max_error(mut self, max_error: u32) -> Self {
        self.max_error = max_error;
        self
    }

    pub fn max_error(&self) -> u32 {
        self.max_error
    }

    pub fn matches(&self, key_id: u64) -> bool {
        self.prefixes
            .iter()
            .any(|prefix| prefix.matches_with_error(key_id, self.max_error))
    }

    pub fn prefixes(&self) -> &[HexPrefix] {
        &self.prefixes
    }

    pub fn normalized_list(&self) -> Vec<String> {
        self.prefixes.iter().map(HexPrefix::normalized).collect()
    }

    // Expected scans before a hit, treating each prefix's matching subset as
    // disjoint. Overlapping prefixes overestimate the match probability and so
    // this returns a lower bound on the true expected count — fine for the
    // progress display, where it's only used to render a percentage.
    pub fn search_space_size(&self) -> u128 {
        let inv_sum: f64 = self
            .prefixes
            .iter()
            .map(|prefix| {
                prefix.match_count(self.max_error) as f64 / prefix.search_space_size() as f64
            })
            .sum();
        if inv_sum <= 0.0 {
            return u128::MAX;
        }
        let approx = 1.0 / inv_sum;
        if approx.is_finite() && approx > 0.0 {
            approx as u128
        } else {
            u128::MAX
        }
    }

    pub(crate) fn pack(&self) -> Vec<u32> {
        let mut packed = Vec::with_capacity(self.prefixes.len() * 4);
        for prefix in &self.prefixes {
            let (mask_high, mask_low) = prefix.mask_words();
            let (value_high, value_low) = prefix.value_words();
            packed.push(mask_high);
            packed.push(mask_low);
            packed.push(value_high);
            packed.push(value_low);
        }
        packed
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FingerprintSearch {
    base_words: [u32; 16],
}

impl FingerprintSearch {
    pub(crate) fn new(public_key: [u8; 32]) -> Self {
        let mut block = [0u8; 64];
        block[0] = 0x99;
        block[1] = 0x00;
        block[2] = 0x33;
        block[3] = 0x04;
        block[8] = 0x16;
        block[9] = 0x09;
        block[10..19].copy_from_slice(&ED25519_OID);
        block[19] = 0x01;
        block[20] = 0x07;
        block[21] = 0x40;
        block[22..54].copy_from_slice(&public_key);
        block[54] = 0x80;
        block[62] = 0x01;
        block[63] = 0xB0;

        let mut base_words = [0u32; 16];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            base_words[index] = u32::from_be_bytes(chunk.try_into().expect("4-byte chunk"));
        }

        Self { base_words }
    }

    pub(crate) fn fingerprint(&self, timestamp: u32) -> [u8; 20] {
        let words = self.sha1_state(timestamp);
        words_to_digest(words)
    }

    pub(crate) fn base_words(&self) -> [u32; 16] {
        self.base_words
    }

    fn sha1_state(&self, timestamp: u32) -> [u32; 5] {
        let mut schedule = [0u32; 80];
        schedule[..16].copy_from_slice(&self.base_words);
        schedule[1] = timestamp;

        for index in 16..80 {
            schedule[index] = (schedule[index - 3]
                ^ schedule[index - 8]
                ^ schedule[index - 14]
                ^ schedule[index - 16])
                .rotate_left(1);
        }

        let [mut a, mut b, mut c, mut d, mut e] = SHA1_INITIAL_STATE;

        for (index, word) in schedule.iter().copied().enumerate() {
            let (f, k) = match index {
                0..=19 => (((b & c) | ((!b) & d)), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => (((b & c) | (b & d) | (c & d)), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };

            let next = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(word);

            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = next;
        }

        [
            SHA1_INITIAL_STATE[0].wrapping_add(a),
            SHA1_INITIAL_STATE[1].wrapping_add(b),
            SHA1_INITIAL_STATE[2].wrapping_add(c),
            SHA1_INITIAL_STATE[3].wrapping_add(d),
            SHA1_INITIAL_STATE[4].wrapping_add(e),
        ]
    }
}

pub fn fingerprint_v4_ed25519(public_key: &[u8; 32], timestamp: u32) -> [u8; 20] {
    FingerprintSearch::new(*public_key).fingerprint(timestamp)
}

pub fn key_id_from_fingerprint(fingerprint: &[u8; 20]) -> u64 {
    let tail: [u8; 8] = fingerprint[12..20].try_into().expect("8-byte key ID tail");
    u64::from_be_bytes(tail)
}

pub fn seed_from_hex(seed_hex: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(seed_hex.trim())?;
    if bytes.len() != 32 {
        bail!("seed must be exactly 32 bytes (64 hex characters)");
    }

    let mut seed = [0u8; 32];
    seed.copy_from_slice(&bytes);
    Ok(seed)
}

pub fn hex_upper<T: AsRef<[u8]>>(bytes: T) -> String {
    hex::encode_upper(bytes)
}

pub fn key_id_hex(key_id: u64) -> String {
    format!("{key_id:016X}")
}

fn decode_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn words_to_digest(words: [u32; 5]) -> [u8; 20] {
    let mut fingerprint = [0u8; 20];
    for (index, word) in words.iter().copied().enumerate() {
        fingerprint[index * 4..(index + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    fingerprint
}

#[cfg(test)]
mod tests {
    use super::{FingerprintSearch, HexPrefix, fingerprint_v4_ed25519, key_id_from_fingerprint};

    // CPU port of the optimized kernel's RUN_CANDIDATE. Mirrors:
    //   * hoisted rounds 0 and 1 (state captured in hr_a/hr_b/hr_e/hr_d_offset)
    //   * precomputed message-expansion constants c_NN[16..=76]
    //   * per-round w1-rotation XOR contributions
    //   * skipping rounds 77, 78, 79 and recovering D_80/E_80 via final rotl-30
    // Returns (key_id_high_word_pre_M, key_id_low_word_pre_M) — i.e. the
    // pre-(+SHA1M_D / +SHA1M_E) live d and e values that the kernel uses to
    // compute key_id_high and key_id_low.
    fn optimized_kernel_d_e(base_words: &[u32; 16], timestamp: u32) -> (u32, u32) {
        let b0 = base_words[0];
        let b2 = base_words[2];
        let b3 = base_words[3];
        let b4 = base_words[4];
        let b5 = base_words[5];
        let b6 = base_words[6];
        let b7 = base_words[7];
        let b8 = base_words[8];
        let b9 = base_words[9];
        let ba = base_words[10];
        let bb = base_words[11];
        let bc = base_words[12];
        let bd = base_words[13];
        let be = base_words[14];
        let bf = base_words[15];

        // Precomputed message-expansion constants (treat w1 = 0).
        let c_16 = (bd ^ b8 ^ b2 ^ b0).rotate_left(1);
        let c_17 = (be ^ b9 ^ b3).rotate_left(1);
        let c_18 = (bf ^ ba ^ b4 ^ b2).rotate_left(1);
        let c_19 = (c_16 ^ bb ^ b5 ^ b3).rotate_left(1);
        let c_20 = (c_17 ^ bc ^ b6 ^ b4).rotate_left(1);
        let c_21 = (c_18 ^ bd ^ b7 ^ b5).rotate_left(1);
        let c_22 = (c_19 ^ be ^ b8 ^ b6).rotate_left(1);
        let c_23 = (c_20 ^ bf ^ b9 ^ b7).rotate_left(1);
        let c_24 = (c_21 ^ c_16 ^ ba ^ b8).rotate_left(1);
        let c_25 = (c_22 ^ c_17 ^ bb ^ b9).rotate_left(1);
        let c_26 = (c_23 ^ c_18 ^ bc ^ ba).rotate_left(1);
        let c_27 = (c_24 ^ c_19 ^ bd ^ bb).rotate_left(1);
        let c_28 = (c_25 ^ c_20 ^ be ^ bc).rotate_left(1);
        let c_29 = (c_26 ^ c_21 ^ bf ^ bd).rotate_left(1);
        let c_30 = (c_27 ^ c_22 ^ c_16 ^ be).rotate_left(1);
        let c_31 = (c_28 ^ c_23 ^ c_17 ^ bf).rotate_left(1);
        let c_32 = (c_29 ^ c_24 ^ c_18 ^ c_16).rotate_left(1);
        let c_33 = (c_30 ^ c_25 ^ c_19 ^ c_17).rotate_left(1);
        let c_34 = (c_31 ^ c_26 ^ c_20 ^ c_18).rotate_left(1);
        let c_35 = (c_32 ^ c_27 ^ c_21 ^ c_19).rotate_left(1);
        let c_36 = (c_33 ^ c_28 ^ c_22 ^ c_20).rotate_left(1);
        let c_37 = (c_34 ^ c_29 ^ c_23 ^ c_21).rotate_left(1);
        let c_38 = (c_35 ^ c_30 ^ c_24 ^ c_22).rotate_left(1);
        let c_39 = (c_36 ^ c_31 ^ c_25 ^ c_23).rotate_left(1);
        let c_40 = (c_37 ^ c_32 ^ c_26 ^ c_24).rotate_left(1);
        let c_41 = (c_38 ^ c_33 ^ c_27 ^ c_25).rotate_left(1);
        let c_42 = (c_39 ^ c_34 ^ c_28 ^ c_26).rotate_left(1);
        let c_43 = (c_40 ^ c_35 ^ c_29 ^ c_27).rotate_left(1);
        let c_44 = (c_41 ^ c_36 ^ c_30 ^ c_28).rotate_left(1);
        let c_45 = (c_42 ^ c_37 ^ c_31 ^ c_29).rotate_left(1);
        let c_46 = (c_43 ^ c_38 ^ c_32 ^ c_30).rotate_left(1);
        let c_47 = (c_44 ^ c_39 ^ c_33 ^ c_31).rotate_left(1);
        let c_48 = (c_45 ^ c_40 ^ c_34 ^ c_32).rotate_left(1);
        let c_49 = (c_46 ^ c_41 ^ c_35 ^ c_33).rotate_left(1);
        let c_50 = (c_47 ^ c_42 ^ c_36 ^ c_34).rotate_left(1);
        let c_51 = (c_48 ^ c_43 ^ c_37 ^ c_35).rotate_left(1);
        let c_52 = (c_49 ^ c_44 ^ c_38 ^ c_36).rotate_left(1);
        let c_53 = (c_50 ^ c_45 ^ c_39 ^ c_37).rotate_left(1);
        let c_54 = (c_51 ^ c_46 ^ c_40 ^ c_38).rotate_left(1);
        let c_55 = (c_52 ^ c_47 ^ c_41 ^ c_39).rotate_left(1);
        let c_56 = (c_53 ^ c_48 ^ c_42 ^ c_40).rotate_left(1);
        let c_57 = (c_54 ^ c_49 ^ c_43 ^ c_41).rotate_left(1);
        let c_58 = (c_55 ^ c_50 ^ c_44 ^ c_42).rotate_left(1);
        let c_59 = (c_56 ^ c_51 ^ c_45 ^ c_43).rotate_left(1);
        let c_60 = (c_57 ^ c_52 ^ c_46 ^ c_44).rotate_left(1);
        let c_61 = (c_58 ^ c_53 ^ c_47 ^ c_45).rotate_left(1);
        let c_62 = (c_59 ^ c_54 ^ c_48 ^ c_46).rotate_left(1);
        let c_63 = (c_60 ^ c_55 ^ c_49 ^ c_47).rotate_left(1);
        let c_64 = (c_61 ^ c_56 ^ c_50 ^ c_48).rotate_left(1);
        let c_65 = (c_62 ^ c_57 ^ c_51 ^ c_49).rotate_left(1);
        let c_66 = (c_63 ^ c_58 ^ c_52 ^ c_50).rotate_left(1);
        let c_67 = (c_64 ^ c_59 ^ c_53 ^ c_51).rotate_left(1);
        let c_68 = (c_65 ^ c_60 ^ c_54 ^ c_52).rotate_left(1);
        let c_69 = (c_66 ^ c_61 ^ c_55 ^ c_53).rotate_left(1);
        let c_70 = (c_67 ^ c_62 ^ c_56 ^ c_54).rotate_left(1);
        let c_71 = (c_68 ^ c_63 ^ c_57 ^ c_55).rotate_left(1);
        let c_72 = (c_69 ^ c_64 ^ c_58 ^ c_56).rotate_left(1);
        let c_73 = (c_70 ^ c_65 ^ c_59 ^ c_57).rotate_left(1);
        let c_74 = (c_71 ^ c_66 ^ c_60 ^ c_58).rotate_left(1);
        let c_75 = (c_72 ^ c_67 ^ c_61 ^ c_59).rotate_left(1);
        let c_76 = (c_73 ^ c_68 ^ c_62 ^ c_60).rotate_left(1);

        // Hoisted round 0 and round 1 outputs.
        let ch = |x: u32, y: u32, z: u32| -> u32 { z ^ (x & (y ^ z)) };
        let parity = |x: u32, y: u32, z: u32| -> u32 { x ^ y ^ z };
        let maj = |x: u32, y: u32, z: u32| -> u32 { (x & y) | (z & (x ^ y)) };

        const SHA1M_A: u32 = 0x6745_2301;
        const SHA1M_B: u32 = 0xEFCD_AB89;
        const SHA1M_C: u32 = 0x98BA_DCFE;
        const SHA1M_D: u32 = 0x1032_5476;
        const SHA1M_E: u32 = 0xC3D2_E1F0;
        const SHA1C00: u32 = 0x5A82_7999;
        const SHA1C01: u32 = 0x6ED9_EBA1;
        const SHA1C02: u32 = 0x8F1B_BCDC;
        const SHA1C03: u32 = 0xCA62_C1D6;

        let hr_b = SHA1M_B.rotate_left(30);
        let hr_e = SHA1M_A
            .rotate_left(5)
            .wrapping_add(ch(SHA1M_B, SHA1M_C, SHA1M_D))
            .wrapping_add(SHA1M_E)
            .wrapping_add(SHA1C00)
            .wrapping_add(b0);
        let hr_a = SHA1M_A.rotate_left(30);
        let hr_d_offset = hr_e
            .rotate_left(5)
            .wrapping_add(ch(SHA1M_A, hr_b, SHA1M_C))
            .wrapping_add(SHA1M_D)
            .wrapping_add(SHA1C00);

        let w1 = timestamp;
        let r = |s: u32| w1.rotate_left(s);

        // SHA-1 step: e_new = rotl(a, 5) + F(b, c, d) + e + K + W; b_new = rotl(b, 30)
        // Inline rotating step that mutates (a, b, c, d, e) by re-binding.
        // For clarity we just use the regular variable order with macro-like
        // helper.
        fn step<F: Fn(u32, u32, u32) -> u32>(
            f: F,
            k: u32,
            a: u32,
            b: u32,
            c: u32,
            d: u32,
            e: u32,
            w: u32,
        ) -> (u32, u32, u32, u32, u32) {
            // Returns the rotated tuple: produces (a', b', c', d', e') after
            // applying one SHA-1 step where a' = new_T, b' = old a,
            // c' = rotl(old b, 30), d' = old c, e' = old d. (Standard SHA-1
            // rotation.)
            let new_a = a
                .rotate_left(5)
                .wrapping_add(f(b, c, d))
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w);
            (new_a, a, b.rotate_left(30), c, d)
        }

        // After hoisted rounds 0 and 1, kernel state per the macro positional
        // convention is (a, b, c, d, e) = (hr_a, hr_b, SHA1M_C, hr_d_offset + w1, hr_e).
        // But the kernel macro represents the SHA-1 state using a rotating
        // mapping (see register-cycling derivation). To run rounds 2..=76 we
        // can match the kernel exactly by treating these as the macro variable
        // bindings and applying steps in the right order — or, equivalently,
        // reconstruct the SHA-1 state (A, B, C, D, E) at end of round 1 and
        // run rounds 2..=76 of standard SHA-1 with the precomputed W values.
        //
        // Standard SHA-1 state at end of round R+1=2 (mapping (B, C, D, E, A)
        // cycle for 5k+1): from my derivation,
        //   variable a after r1 = B_2 ?  Let me derive.
        //   End of round 0 (R+1=1, ≡ 1 mod 5): (a,b,c,d,e) = (B1, C1, D1, E1, A1).
        //   End of round 1 (R+1=2, ≡ 2 mod 5): (a,b,c,d,e) = (C2, D2, E2, A2, B2).
        // So at end of round 1: SHA-1 state (A2, B2, C2, D2, E2) maps to
        //   A2 = variable d = hr_d_offset + w1
        //   B2 = variable e = hr_e
        //   C2 = variable a = hr_a
        //   D2 = variable b = hr_b
        //   E2 = variable c = SHA1M_C
        let mut state_a = hr_d_offset.wrapping_add(w1);
        let mut state_b = hr_e;
        let mut state_c = hr_a;
        let mut state_d = hr_b;
        let mut state_e = SHA1M_C;

        // Per-round W with w1 contributions (matches kernel rounds 2..=76).
        let w_round = [
            b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf, // 2..=15
            c_16,                                                // 16
            c_17 ^ r(1),                                         // 17
            c_18,                                                // 18
            c_19,                                                // 19
            c_20 ^ r(2),                                         // 20
            c_21,                                                // 21
            c_22,                                                // 22
            c_23 ^ r(3),                                         // 23
            c_24,                                                // 24
            c_25 ^ r(2),                                         // 25
            c_26 ^ r(4),                                         // 26
            c_27,                                                // 27
            c_28,                                                // 28
            c_29 ^ r(5),                                         // 29
            c_30,                                                // 30
            c_31 ^ r(2) ^ r(4),                                  // 31
            c_32 ^ r(6),                                         // 32
            c_33 ^ r(2) ^ r(3),                                  // 33
            c_34,                                                // 34
            c_35 ^ r(7),                                         // 35
            c_36 ^ r(4),                                         // 36
            c_37 ^ r(4) ^ r(6),                                  // 37
            c_38 ^ r(8),                                         // 38
            c_39 ^ r(4),                                         // 39
            c_40,                                                // 40
            c_41 ^ r(4) ^ r(9),                                  // 41
            c_42,                                                // 42
            c_43 ^ r(6) ^ r(8),                                  // 43
            c_44 ^ r(10),                                        // 44
            c_45 ^ r(3) ^ r(6) ^ r(7),                           // 45
            c_46,                                                // 46
            c_47 ^ r(4) ^ r(11),                                 // 47
            c_48 ^ r(4) ^ r(8),                                  // 48
            c_49 ^ r(3) ^ r(4) ^ r(8) ^ r(5) ^ r(10),            // 49
            c_50 ^ r(12),                                        // 50
            c_51 ^ r(8),                                         // 51
            c_52 ^ r(4) ^ r(6),                                  // 52
            c_53 ^ r(4) ^ r(8) ^ r(13),                          // 53
            c_54,                                                // 54
            c_55 ^ r(7) ^ r(10) ^ r(12),                         // 55
            c_56 ^ r(14),                                        // 56
            c_57 ^ r(4) ^ r(6) ^ r(7) ^ r(10) ^ r(11),           // 57
            c_58 ^ r(8),                                         // 58
            c_59 ^ r(4) ^ r(8) ^ r(15),                          // 59
            c_60 ^ r(8) ^ r(12),                                 // 60
            c_61 ^ r(4) ^ r(7) ^ r(8) ^ r(12) ^ r(14),           // 61
            c_62 ^ r(16),                                        // 62
            c_63 ^ r(4) ^ r(6) ^ r(8) ^ r(12),                   // 63
            c_64 ^ r(8),                                         // 64
            c_65 ^ r(4) ^ r(6) ^ r(7) ^ r(8) ^ r(12) ^ r(17),    // 65
            c_66,                                                // 66
            c_67 ^ r(14) ^ r(16),                                // 67
            c_68 ^ r(8) ^ r(18),                                 // 68
            c_69 ^ r(11) ^ r(14) ^ r(15),                        // 69
            c_70,                                                // 70
            c_71 ^ r(12) ^ r(19),                                // 71
            c_72 ^ r(12) ^ r(16),                                // 72
            c_73 ^ r(5) ^ r(11) ^ r(12) ^ r(13) ^ r(16) ^ r(18), // 73
            c_74 ^ r(20),                                        // 74
            c_75 ^ r(8) ^ r(16),                                 // 75
            c_76 ^ r(6) ^ r(12) ^ r(14),                         // 76
        ];

        for (offset, &w) in w_round.iter().enumerate() {
            let round = offset + 2;
            let (f, k): (fn(u32, u32, u32) -> u32, u32) = match round {
                0..=19 => (ch, SHA1C00),
                20..=39 => (parity, SHA1C01),
                40..=59 => (maj, SHA1C02),
                _ => (parity, SHA1C03),
            };
            let next = state_a
                .rotate_left(5)
                .wrapping_add(f(state_b, state_c, state_d))
                .wrapping_add(state_e)
                .wrapping_add(k)
                .wrapping_add(w);
            let _ = step::<fn(u32, u32, u32) -> u32>; // silence unused warning
            state_e = state_d;
            state_d = state_c;
            state_c = state_b.rotate_left(30);
            state_b = state_a;
            state_a = next;
        }

        // After round 76: SHA-1 state is (A_77, B_77, C_77, D_77, E_77).
        // The kernel recovers D_80 = rotl(T_76, 30) and E_80 = rotl(T_75, 30).
        // T_76 = A_77 = state_a. T_75 = A_76 = B_77 = state_b.
        let d_80 = state_a.rotate_left(30);
        let e_80 = state_b.rotate_left(30);
        (d_80, e_80)
    }

    #[test]
    fn optimized_kernel_matches_reference_sha1() {
        let public_key = [0x42; 32];
        let search = FingerprintSearch::new(public_key);
        let base_words = search.base_words();
        for timestamp in [
            0u32,
            1,
            0x1357_9BDF,
            0x8000_0000,
            0xFFFF_FFFF,
            0xDEAD_BEEF,
            0xCAFE_F00D,
            0x4242_4242,
        ] {
            let reference = search.sha1_state(timestamp);
            let ref_d = reference[3].wrapping_sub(0x1032_5476);
            let ref_e = reference[4].wrapping_sub(0xC3D2_E1F0);
            let (opt_d, opt_e) = optimized_kernel_d_e(&base_words, timestamp);
            assert_eq!(
                ref_d, opt_d,
                "D mismatch at timestamp 0x{timestamp:08X}: \
                 reference={ref_d:08X} optimized={opt_d:08X}"
            );
            assert_eq!(
                ref_e, opt_e,
                "E mismatch at timestamp 0x{timestamp:08X}: \
                 reference={ref_e:08X} optimized={opt_e:08X}"
            );
        }
    }

    // Emulates the kernel's exact macro instruction sequence, including which
    // rounds use SHA1_STEP (rotates b) vs SHA1_STEP_FINAL (skips the dead
    // b-rotation). This catches the kind of bug where the wrong step variant
    // is used and downstream rounds read a stale variable.
    //
    // It is critical that this exactly mirrors RUN_CANDIDATE in
    // src/gpu_kernel.hip — when the kernel changes which rounds use FINAL,
    // this function must update too.
    fn kernel_emulated_d_e(base_words: &[u32; 16], timestamp: u32) -> (u32, u32) {
        const SHA1M_A: u32 = 0x6745_2301;
        const SHA1M_B: u32 = 0xEFCD_AB89;
        const SHA1M_C: u32 = 0x98BA_DCFE;
        const SHA1M_D: u32 = 0x1032_5476;
        const SHA1M_E: u32 = 0xC3D2_E1F0;
        const SHA1C00: u32 = 0x5A82_7999;
        const SHA1C01: u32 = 0x6ED9_EBA1;
        const SHA1C02: u32 = 0x8F1B_BCDC;
        const SHA1C03: u32 = 0xCA62_C1D6;

        let ch = |x: u32, y: u32, z: u32| -> u32 { z ^ (x & (y ^ z)) };
        let parity = |x: u32, y: u32, z: u32| -> u32 { x ^ y ^ z };
        let maj = |x: u32, y: u32, z: u32| -> u32 { (x & y) | (z & (x ^ y)) };

        // Macro variable order: (a, b, c, d, e). After each step, the 5th arg
        // (e) gets the new T, and (if `rotate_b`) the 2nd arg (b) gets
        // rotated by 30.
        //
        // The kernel's macro invocations pass these positionally; we mirror
        // that by always tracking the kernel's variables [a, b, c, d, e] and
        // updating index `idx_e` (the 5th positional arg) and rotating index
        // `idx_b` (the 2nd positional arg) when `rotate_b` is true.
        fn run_step(
            vars: &mut [u32; 5],
            f: fn(u32, u32, u32) -> u32,
            k: u32,
            indices: [usize; 5], // [a, b, c, d, e] kernel variable indices passed to macro
            w: u32,
            rotate_b: bool,
        ) {
            let va = vars[indices[0]];
            let vb = vars[indices[1]];
            let vc = vars[indices[2]];
            let vd = vars[indices[3]];
            let ve = vars[indices[4]];
            let new_e = va
                .rotate_left(5)
                .wrapping_add(f(vb, vc, vd))
                .wrapping_add(ve)
                .wrapping_add(k)
                .wrapping_add(w);
            vars[indices[4]] = new_e;
            if rotate_b {
                vars[indices[1]] = vb.rotate_left(30);
            }
        }

        // Index mapping: 0=a, 1=b, 2=c, 3=d, 4=e.
        let a = 0usize;
        let b = 1usize;
        let c = 2usize;
        let d = 3usize;
        let e = 4usize;

        // Hoisted round 0 and round 1 state, matching kernel.
        let b0 = base_words[0];
        let hr_b = SHA1M_B.rotate_left(30);
        let hr_e = SHA1M_A
            .rotate_left(5)
            .wrapping_add(ch(SHA1M_B, SHA1M_C, SHA1M_D))
            .wrapping_add(SHA1M_E)
            .wrapping_add(SHA1C00)
            .wrapping_add(b0);
        let hr_a = SHA1M_A.rotate_left(30);
        let hr_d_offset = hr_e
            .rotate_left(5)
            .wrapping_add(ch(SHA1M_A, hr_b, SHA1M_C))
            .wrapping_add(SHA1M_D)
            .wrapping_add(SHA1C00);

        let w1 = timestamp;
        let mut vars = [hr_a, hr_b, SHA1M_C, hr_d_offset.wrapping_add(w1), hr_e];

        let r = |s: u32| w1.rotate_left(s);
        let w1s04_06 = r(4) ^ r(6);
        let w1s04_08 = r(4) ^ r(8);
        let w1s08_12 = r(8) ^ r(12);
        let w1s04_06_07 = w1s04_06 ^ r(7);

        // Rounds 2..=15 with kernel-base words.
        let base_w: [u32; 14] = [
            base_words[2],
            base_words[3],
            base_words[4],
            base_words[5],
            base_words[6],
            base_words[7],
            base_words[8],
            base_words[9],
            base_words[10],
            base_words[11],
            base_words[12],
            base_words[13],
            base_words[14],
            base_words[15],
        ];
        // The macro positional patterns for rounds 2..15 from the kernel.
        let r2_15_indices: [[usize; 5]; 14] = [
            [d, e, a, b, c], // round 2
            [c, d, e, a, b], // round 3
            [b, c, d, e, a], // round 4
            [a, b, c, d, e], // round 5
            [e, a, b, c, d], // round 6
            [d, e, a, b, c], // round 7
            [c, d, e, a, b], // round 8
            [b, c, d, e, a], // round 9
            [a, b, c, d, e], // round 10
            [e, a, b, c, d], // round 11
            [d, e, a, b, c], // round 12
            [c, d, e, a, b], // round 13
            [b, c, d, e, a], // round 14
            [a, b, c, d, e], // round 15
        ];
        for (i, idx) in r2_15_indices.iter().enumerate() {
            run_step(&mut vars, ch, SHA1C00, *idx, base_w[i], true);
        }

        // Compute c_NN constants like the kernel does (needed to express
        // round-16+ message words).
        let b2 = base_words[2];
        let b3 = base_words[3];
        let b4 = base_words[4];
        let b5 = base_words[5];
        let b6 = base_words[6];
        let b7 = base_words[7];
        let b8 = base_words[8];
        let b9 = base_words[9];
        let ba = base_words[10];
        let bb = base_words[11];
        let bc = base_words[12];
        let bd = base_words[13];
        let be = base_words[14];
        let bf = base_words[15];

        let c_16 = (bd ^ b8 ^ b2 ^ b0).rotate_left(1);
        let c_17 = (be ^ b9 ^ b3).rotate_left(1);
        let c_18 = (bf ^ ba ^ b4 ^ b2).rotate_left(1);
        let c_19 = (c_16 ^ bb ^ b5 ^ b3).rotate_left(1);
        let c_20 = (c_17 ^ bc ^ b6 ^ b4).rotate_left(1);
        let c_21 = (c_18 ^ bd ^ b7 ^ b5).rotate_left(1);
        let c_22 = (c_19 ^ be ^ b8 ^ b6).rotate_left(1);
        let c_23 = (c_20 ^ bf ^ b9 ^ b7).rotate_left(1);
        let c_24 = (c_21 ^ c_16 ^ ba ^ b8).rotate_left(1);
        let c_25 = (c_22 ^ c_17 ^ bb ^ b9).rotate_left(1);
        let c_26 = (c_23 ^ c_18 ^ bc ^ ba).rotate_left(1);
        let c_27 = (c_24 ^ c_19 ^ bd ^ bb).rotate_left(1);
        let c_28 = (c_25 ^ c_20 ^ be ^ bc).rotate_left(1);
        let c_29 = (c_26 ^ c_21 ^ bf ^ bd).rotate_left(1);
        let c_30 = (c_27 ^ c_22 ^ c_16 ^ be).rotate_left(1);
        let c_31 = (c_28 ^ c_23 ^ c_17 ^ bf).rotate_left(1);
        let c_32 = (c_29 ^ c_24 ^ c_18 ^ c_16).rotate_left(1);
        let c_33 = (c_30 ^ c_25 ^ c_19 ^ c_17).rotate_left(1);
        let c_34 = (c_31 ^ c_26 ^ c_20 ^ c_18).rotate_left(1);
        let c_35 = (c_32 ^ c_27 ^ c_21 ^ c_19).rotate_left(1);
        let c_36 = (c_33 ^ c_28 ^ c_22 ^ c_20).rotate_left(1);
        let c_37 = (c_34 ^ c_29 ^ c_23 ^ c_21).rotate_left(1);
        let c_38 = (c_35 ^ c_30 ^ c_24 ^ c_22).rotate_left(1);
        let c_39 = (c_36 ^ c_31 ^ c_25 ^ c_23).rotate_left(1);
        let c_40 = (c_37 ^ c_32 ^ c_26 ^ c_24).rotate_left(1);
        let c_41 = (c_38 ^ c_33 ^ c_27 ^ c_25).rotate_left(1);
        let c_42 = (c_39 ^ c_34 ^ c_28 ^ c_26).rotate_left(1);
        let c_43 = (c_40 ^ c_35 ^ c_29 ^ c_27).rotate_left(1);
        let c_44 = (c_41 ^ c_36 ^ c_30 ^ c_28).rotate_left(1);
        let c_45 = (c_42 ^ c_37 ^ c_31 ^ c_29).rotate_left(1);
        let c_46 = (c_43 ^ c_38 ^ c_32 ^ c_30).rotate_left(1);
        let c_47 = (c_44 ^ c_39 ^ c_33 ^ c_31).rotate_left(1);
        let c_48 = (c_45 ^ c_40 ^ c_34 ^ c_32).rotate_left(1);
        let c_49 = (c_46 ^ c_41 ^ c_35 ^ c_33).rotate_left(1);
        let c_50 = (c_47 ^ c_42 ^ c_36 ^ c_34).rotate_left(1);
        let c_51 = (c_48 ^ c_43 ^ c_37 ^ c_35).rotate_left(1);
        let c_52 = (c_49 ^ c_44 ^ c_38 ^ c_36).rotate_left(1);
        let c_53 = (c_50 ^ c_45 ^ c_39 ^ c_37).rotate_left(1);
        let c_54 = (c_51 ^ c_46 ^ c_40 ^ c_38).rotate_left(1);
        let c_55 = (c_52 ^ c_47 ^ c_41 ^ c_39).rotate_left(1);
        let c_56 = (c_53 ^ c_48 ^ c_42 ^ c_40).rotate_left(1);
        let c_57 = (c_54 ^ c_49 ^ c_43 ^ c_41).rotate_left(1);
        let c_58 = (c_55 ^ c_50 ^ c_44 ^ c_42).rotate_left(1);
        let c_59 = (c_56 ^ c_51 ^ c_45 ^ c_43).rotate_left(1);
        let c_60 = (c_57 ^ c_52 ^ c_46 ^ c_44).rotate_left(1);
        let c_61 = (c_58 ^ c_53 ^ c_47 ^ c_45).rotate_left(1);
        let c_62 = (c_59 ^ c_54 ^ c_48 ^ c_46).rotate_left(1);
        let c_63 = (c_60 ^ c_55 ^ c_49 ^ c_47).rotate_left(1);
        let c_64 = (c_61 ^ c_56 ^ c_50 ^ c_48).rotate_left(1);
        let c_65 = (c_62 ^ c_57 ^ c_51 ^ c_49).rotate_left(1);
        let c_66 = (c_63 ^ c_58 ^ c_52 ^ c_50).rotate_left(1);
        let c_67 = (c_64 ^ c_59 ^ c_53 ^ c_51).rotate_left(1);
        let c_68 = (c_65 ^ c_60 ^ c_54 ^ c_52).rotate_left(1);
        let c_69 = (c_66 ^ c_61 ^ c_55 ^ c_53).rotate_left(1);
        let c_70 = (c_67 ^ c_62 ^ c_56 ^ c_54).rotate_left(1);
        let c_71 = (c_68 ^ c_63 ^ c_57 ^ c_55).rotate_left(1);
        let c_72 = (c_69 ^ c_64 ^ c_58 ^ c_56).rotate_left(1);
        let c_73 = (c_70 ^ c_65 ^ c_59 ^ c_57).rotate_left(1);
        let c_74 = (c_71 ^ c_66 ^ c_60 ^ c_58).rotate_left(1);
        let c_75 = (c_72 ^ c_67 ^ c_61 ^ c_59).rotate_left(1);
        let c_76 = (c_73 ^ c_68 ^ c_62 ^ c_60).rotate_left(1);

        // Rounds 16..=19 (F0).
        run_step(&mut vars, ch, SHA1C00, [e, a, b, c, d], c_16, true);
        run_step(&mut vars, ch, SHA1C00, [d, e, a, b, c], c_17 ^ r(1), true);
        run_step(&mut vars, ch, SHA1C00, [c, d, e, a, b], c_18, true);
        run_step(&mut vars, ch, SHA1C00, [b, c, d, e, a], c_19, true);

        // Rounds 20..=39 (F1).
        run_step(&mut vars, parity, SHA1C01, [a, b, c, d, e], c_20 ^ r(2), true);
        run_step(&mut vars, parity, SHA1C01, [e, a, b, c, d], c_21, true);
        run_step(&mut vars, parity, SHA1C01, [d, e, a, b, c], c_22, true);
        run_step(&mut vars, parity, SHA1C01, [c, d, e, a, b], c_23 ^ r(3), true);
        run_step(&mut vars, parity, SHA1C01, [b, c, d, e, a], c_24, true);
        run_step(&mut vars, parity, SHA1C01, [a, b, c, d, e], c_25 ^ r(2), true);
        run_step(&mut vars, parity, SHA1C01, [e, a, b, c, d], c_26 ^ r(4), true);
        run_step(&mut vars, parity, SHA1C01, [d, e, a, b, c], c_27, true);
        run_step(&mut vars, parity, SHA1C01, [c, d, e, a, b], c_28, true);
        run_step(&mut vars, parity, SHA1C01, [b, c, d, e, a], c_29 ^ r(5), true);
        run_step(&mut vars, parity, SHA1C01, [a, b, c, d, e], c_30, true);
        run_step(&mut vars, parity, SHA1C01, [e, a, b, c, d], c_31 ^ r(2) ^ r(4), true);
        run_step(&mut vars, parity, SHA1C01, [d, e, a, b, c], c_32 ^ r(6), true);
        run_step(&mut vars, parity, SHA1C01, [c, d, e, a, b], c_33 ^ r(2) ^ r(3), true);
        run_step(&mut vars, parity, SHA1C01, [b, c, d, e, a], c_34, true);
        run_step(&mut vars, parity, SHA1C01, [a, b, c, d, e], c_35 ^ r(7), true);
        run_step(&mut vars, parity, SHA1C01, [e, a, b, c, d], c_36 ^ r(4), true);
        run_step(&mut vars, parity, SHA1C01, [d, e, a, b, c], c_37 ^ w1s04_06, true);
        run_step(&mut vars, parity, SHA1C01, [c, d, e, a, b], c_38 ^ r(8), true);
        run_step(&mut vars, parity, SHA1C01, [b, c, d, e, a], c_39 ^ r(4), true);

        // Rounds 40..=59 (F2).
        run_step(&mut vars, maj, SHA1C02, [a, b, c, d, e], c_40, true);
        run_step(&mut vars, maj, SHA1C02, [e, a, b, c, d], c_41 ^ r(4) ^ r(9), true);
        run_step(&mut vars, maj, SHA1C02, [d, e, a, b, c], c_42, true);
        run_step(&mut vars, maj, SHA1C02, [c, d, e, a, b], c_43 ^ r(6) ^ r(8), true);
        run_step(&mut vars, maj, SHA1C02, [b, c, d, e, a], c_44 ^ r(10), true);
        run_step(&mut vars, maj, SHA1C02, [a, b, c, d, e], c_45 ^ r(3) ^ r(6) ^ r(7), true);
        run_step(&mut vars, maj, SHA1C02, [e, a, b, c, d], c_46, true);
        run_step(&mut vars, maj, SHA1C02, [d, e, a, b, c], c_47 ^ r(4) ^ r(11), true);
        run_step(&mut vars, maj, SHA1C02, [c, d, e, a, b], c_48 ^ w1s04_08, true);
        run_step(
            &mut vars,
            maj,
            SHA1C02,
            [b, c, d, e, a],
            c_49 ^ r(3) ^ w1s04_08 ^ r(5) ^ r(10),
            true,
        );
        run_step(&mut vars, maj, SHA1C02, [a, b, c, d, e], c_50 ^ r(12), true);
        run_step(&mut vars, maj, SHA1C02, [e, a, b, c, d], c_51 ^ r(8), true);
        run_step(&mut vars, maj, SHA1C02, [d, e, a, b, c], c_52 ^ w1s04_06, true);
        run_step(&mut vars, maj, SHA1C02, [c, d, e, a, b], c_53 ^ w1s04_08 ^ r(13), true);
        run_step(&mut vars, maj, SHA1C02, [b, c, d, e, a], c_54, true);
        run_step(&mut vars, maj, SHA1C02, [a, b, c, d, e], c_55 ^ r(7) ^ r(10) ^ r(12), true);
        run_step(&mut vars, maj, SHA1C02, [e, a, b, c, d], c_56 ^ r(14), true);
        run_step(
            &mut vars,
            maj,
            SHA1C02,
            [d, e, a, b, c],
            c_57 ^ w1s04_06_07 ^ r(10) ^ r(11),
            true,
        );
        run_step(&mut vars, maj, SHA1C02, [c, d, e, a, b], c_58 ^ r(8), true);
        run_step(&mut vars, maj, SHA1C02, [b, c, d, e, a], c_59 ^ w1s04_08 ^ r(15), true);

        // Rounds 60..=76 (F1).
        run_step(&mut vars, parity, SHA1C03, [a, b, c, d, e], c_60 ^ w1s08_12, true);
        run_step(
            &mut vars,
            parity,
            SHA1C03,
            [e, a, b, c, d],
            c_61 ^ r(4) ^ r(7) ^ w1s08_12 ^ r(14),
            true,
        );
        run_step(&mut vars, parity, SHA1C03, [d, e, a, b, c], c_62 ^ r(16), true);
        run_step(&mut vars, parity, SHA1C03, [c, d, e, a, b], c_63 ^ w1s04_06 ^ w1s08_12, true);
        run_step(&mut vars, parity, SHA1C03, [b, c, d, e, a], c_64 ^ r(8), true);
        run_step(
            &mut vars,
            parity,
            SHA1C03,
            [a, b, c, d, e],
            c_65 ^ w1s04_06_07 ^ w1s08_12 ^ r(17),
            true,
        );
        run_step(&mut vars, parity, SHA1C03, [e, a, b, c, d], c_66, true);
        run_step(&mut vars, parity, SHA1C03, [d, e, a, b, c], c_67 ^ r(14) ^ r(16), true);
        run_step(&mut vars, parity, SHA1C03, [c, d, e, a, b], c_68 ^ r(8) ^ r(18), true);
        run_step(&mut vars, parity, SHA1C03, [b, c, d, e, a], c_69 ^ r(11) ^ r(14) ^ r(15), true);
        run_step(&mut vars, parity, SHA1C03, [a, b, c, d, e], c_70, true);
        run_step(&mut vars, parity, SHA1C03, [e, a, b, c, d], c_71 ^ r(12) ^ r(19), true);
        run_step(&mut vars, parity, SHA1C03, [d, e, a, b, c], c_72 ^ r(12) ^ r(16), true);
        run_step(
            &mut vars,
            parity,
            SHA1C03,
            [c, d, e, a, b],
            c_73 ^ r(5) ^ r(11) ^ r(12) ^ r(13) ^ r(16) ^ r(18),
            true,
        );
        run_step(&mut vars, parity, SHA1C03, [b, c, d, e, a], c_74 ^ r(20), true);
        // Round 75: full STEP (round 76 needs the rotated b).
        run_step(&mut vars, parity, SHA1C03, [a, b, c, d, e], c_75 ^ r(8) ^ r(16), true);
        // Round 76: STEP_FINAL (no b-rotate; variable a is dead afterwards).
        run_step(
            &mut vars,
            parity,
            SHA1C03,
            [e, a, b, c, d],
            c_76 ^ r(6) ^ r(12) ^ r(14),
            false,
        );

        // Final rotations to recover D_80 and E_80.
        (vars[d].rotate_left(30), vars[e].rotate_left(30))
    }

    #[test]
    fn kernel_emulation_matches_reference_sha1() {
        let public_key = [0x42; 32];
        let search = FingerprintSearch::new(public_key);
        let base_words = search.base_words();
        for timestamp in [
            0u32,
            1,
            0x1357_9BDF,
            0x8000_0000,
            0xFFFF_FFFF,
            0xDEAD_BEEF,
            0xCAFE_F00D,
            0x4242_4242,
        ] {
            let reference = search.sha1_state(timestamp);
            let ref_d = reference[3].wrapping_sub(0x1032_5476);
            let ref_e = reference[4].wrapping_sub(0xC3D2_E1F0);
            let (kern_d, kern_e) = kernel_emulated_d_e(&base_words, timestamp);
            assert_eq!(
                ref_d, kern_d,
                "kernel-emulated D mismatch at timestamp 0x{timestamp:08X}: \
                 reference={ref_d:08X} kernel={kern_d:08X}"
            );
            assert_eq!(
                ref_e, kern_e,
                "kernel-emulated E mismatch at timestamp 0x{timestamp:08X}: \
                 reference={ref_e:08X} kernel={kern_e:08X}"
            );
        }
    }

    #[test]
    fn error_count_counts_wrong_nibbles() {
        let prefix = HexPrefix::parse("AAAA").expect("valid prefix");
        // Exact match.
        assert_eq!(prefix.error_count(0xAAAA_0000_0000_0000), 0);
        // One wrong nibble (low nibble of the prefix region).
        assert_eq!(prefix.error_count(0xAAA0_0000_0000_0000), 1);
        // One wrong nibble in the middle.
        assert_eq!(prefix.error_count(0xABAA_0000_0000_0000), 1);
        // Two wrong nibbles.
        assert_eq!(prefix.error_count(0xAB0A_0000_0000_0000), 2);
        // Bits outside the prefix region are ignored.
        assert_eq!(prefix.error_count(0xAAAA_FFFF_FFFF_FFFF), 0);
    }

    #[test]
    fn matches_with_error_honors_threshold() {
        let set = super::HexPrefixSet::parse(["AAAA"])
            .expect("valid set")
            .with_max_error(1);
        assert!(set.matches(0xAAA0_0000_0000_0000));
        assert!(set.matches(0xABAA_0000_0000_0000));
        assert!(set.matches(0xAAAA_0000_0000_0000));
        assert!(!set.matches(0xAB0A_0000_0000_0000));
    }

    #[test]
    fn match_count_matches_combinatorial_formula() {
        let prefix = HexPrefix::parse("AAAA").expect("valid prefix");
        // n = 4 nibbles.
        assert_eq!(prefix.match_count(0), 1); // C(4,0)*15^0
        assert_eq!(prefix.match_count(1), 1 + 4 * 15); // + C(4,1)*15
        assert_eq!(prefix.match_count(2), 1 + 4 * 15 + 6 * 225);
        // e capped at n: max_error beyond n matches the entire space (16^4).
        assert_eq!(prefix.match_count(10), 1u128 << 16);
    }

    #[test]
    fn odd_length_prefix_matches_high_nibble() {
        let prefix = HexPrefix::parse("abc").expect("valid prefix");
        assert!(prefix.matches(0xABC0_0000_0000_0000));
        assert!(prefix.matches(0xABCF_FFFF_FFFF_FFFF));
        assert!(!prefix.matches(0xABD0_0000_0000_0000));
    }

    #[test]
    fn fingerprint_key_id_tail_matches_helper() {
        let public_key = [0x11; 32];
        let timestamp = 0x1234_5678;
        let fingerprint = fingerprint_v4_ed25519(&public_key, timestamp);
        let search = FingerprintSearch::new(public_key);
        assert_eq!(search.fingerprint(timestamp), fingerprint);
        assert_eq!(
            key_id_from_fingerprint(&fingerprint),
            key_id_from_fingerprint(&search.fingerprint(timestamp))
        );
    }
}
