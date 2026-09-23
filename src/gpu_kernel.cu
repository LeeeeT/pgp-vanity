/* Shared GPU kernel source, compiled at runtime by HIPRTC (HIP) or
   NVRTC (CUDA). Keep the code within the common subset of both dialects and
   guard backend-specific intrinsics with preprocessor checks. */

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

__device__ inline __attribute__((always_inline)) unsigned int rotl32(
    unsigned int value,
    unsigned int amount)
{
#if __has_builtin(__builtin_rotateleft32)
    return __builtin_rotateleft32(value, amount);
#else
    return (value << amount) | (value >> (32u - amount));
#endif
}

__device__ inline __attribute__((always_inline)) unsigned int sha1_ch(
    unsigned int x,
    unsigned int y,
    unsigned int z)
{
#if defined(__AMDGCN__)
    unsigned int result;
    // LLVM otherwise lowers choose to two instructions (xor + and-or).
    asm("v_bfi_b32 %0, %1, %2, %3"
        : "=v"(result) : "v"(x), "v"(y), "v"(z));
    return result;
#else
    return z ^ (x & (y ^ z));
#endif
}

__device__ inline __attribute__((always_inline)) unsigned int sha1_parity(
    unsigned int x,
    unsigned int y,
    unsigned int z)
{
    return x ^ y ^ z;
}

__device__ inline __attribute__((always_inline)) unsigned int sha1_maj(
    unsigned int x,
    unsigned int y,
    unsigned int z)
{
#if defined(__AMDGCN__)
    unsigned int result;
    // If x == z, choose z; otherwise choose y: xor + bit-select.
    asm("v_bfi_b32 %0, %1, %2, %3"
        : "=v"(result) : "v"(x ^ z), "v"(y), "v"(z));
    return result;
#else
    return (x & y) | (z & (x ^ y));
#endif
}

__device__ inline __attribute__((always_inline)) unsigned int load_best_timestamp(
    volatile unsigned int* best_timestamp)
{
    unsigned int best_so_far = 0xFFFFFFFFu;

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        best_so_far = *best_timestamp;
    }

#if __has_builtin(__builtin_amdgcn_readfirstlane)
    return __builtin_amdgcn_readfirstlane(best_so_far);
#elif defined(__CUDACC_RTC__) || defined(__CUDA_ARCH__)
    return __shfl_sync(0xFFFFFFFFu, best_so_far, 0);
#else
    return best_so_far;
#endif
}

#define SHA1C00 0x5A827999u
#define SHA1C01 0x6ED9EBA1u
#define SHA1C02 0x8F1BBCDCu
#define SHA1C03 0xCA62C1D6u
#define SHA1M_A 0x67452301u
#define SHA1M_B 0xEFCDAB89u
#define SHA1M_C 0x98BADCFEu
#define SHA1M_D 0x10325476u
#define SHA1M_E 0xC3D2E1F0u
#define NOT_FOUND_TIMESTAMP 0xFFFFFFFFu
#ifndef POLL_INTERVAL
#define POLL_INTERVAL 4u
#endif
#ifndef FULL_BATCH_OUTER_LOOPS
#define FULL_BATCH_OUTER_LOOPS 128u
#endif
#ifndef MATCH_POLL_GROUPS
#define MATCH_POLL_GROUPS 8u
#endif

#define SHA1_F0 sha1_ch
#define SHA1_F1 sha1_parity
#define SHA1_F2 sha1_maj

// Standard step: adds K to message word x at runtime.
#define SHA1_STEP(F, K, a, b, c, d, e, x) do { \
    (e) = rotl32((a), 5u) + F((b), (c), (d)) + (e) + (K) + (x); \
    (b) = rotl32((b), 30u); \
} while (0)

// Optimized step for rounds whose message word collapses to a constant: the
// caller supplies `xk` already including the round constant K.
#define SHA1_STEP_K(F, a, b, c, d, e, xk) do { \
    (e) = rotl32((a), 5u) + F((b), (c), (d)) + (e) + (xk); \
    (b) = rotl32((b), 30u); \
} while (0)

// Final step variant: skip the b = rotl(b, 30) since the result is dead.
// Used at the last "real" round (76 here) where only the new-T value matters
// and the b-rotation is discarded.
#define SHA1_STEP_FINAL(F, K, a, b, c, d, e, x) do { \
    (e) = rotl32((a), 5u) + F((b), (c), (d)) + (e) + (K) + (x); \
} while (0)

// Precompute the message expansion w[16..79] in two parts:
//   * c_NNs: the part that depends only on the constant base words (treating
//     the timestamp word w1 as zero). Computed once per thread.
//   * The contribution from w1 is added per-candidate as a XOR of rotated
//     copies of w1 (see RUN_CANDIDATE).
//
// Rounds 77, 78, 79 are skipped (we recover D_80 = rotl(T_76, 30) and
// E_80 = rotl(T_75, 30) directly), so c_77s..c_79s and their derived
// constants are not needed.
//
// Rounds whose w[i] has zero w1 contribution are pre-folded with the round
// constant K into c_NNsK to save one addition inside the candidate loop.
#define DECLARE_EXPANSION_CONSTANTS \
    const unsigned int c_16s = rotl32((based ^ base8 ^ base2 ^ base0), 1u); \
    const unsigned int c_17s = rotl32((basee ^ base9 ^ base3        ), 1u); \
    const unsigned int c_18s = rotl32((basef ^ basea ^ base4 ^ base2), 1u); \
    const unsigned int c_19s = rotl32((c_16s ^ baseb ^ base5 ^ base3), 1u); \
    const unsigned int c_20s = rotl32((c_17s ^ basec ^ base6 ^ base4), 1u); \
    const unsigned int c_21s = rotl32((c_18s ^ based ^ base7 ^ base5), 1u); \
    const unsigned int c_22s = rotl32((c_19s ^ basee ^ base8 ^ base6), 1u); \
    const unsigned int c_23s = rotl32((c_20s ^ basef ^ base9 ^ base7), 1u); \
    const unsigned int c_24s = rotl32((c_21s ^ c_16s ^ basea ^ base8), 1u); \
    const unsigned int c_25s = rotl32((c_22s ^ c_17s ^ baseb ^ base9), 1u); \
    const unsigned int c_26s = rotl32((c_23s ^ c_18s ^ basec ^ basea), 1u); \
    const unsigned int c_27s = rotl32((c_24s ^ c_19s ^ based ^ baseb), 1u); \
    const unsigned int c_28s = rotl32((c_25s ^ c_20s ^ basee ^ basec), 1u); \
    const unsigned int c_29s = rotl32((c_26s ^ c_21s ^ basef ^ based), 1u); \
    const unsigned int c_30s = rotl32((c_27s ^ c_22s ^ c_16s ^ basee), 1u); \
    const unsigned int c_31s = rotl32((c_28s ^ c_23s ^ c_17s ^ basef), 1u); \
    const unsigned int c_32s = rotl32((c_29s ^ c_24s ^ c_18s ^ c_16s), 1u); \
    const unsigned int c_33s = rotl32((c_30s ^ c_25s ^ c_19s ^ c_17s), 1u); \
    const unsigned int c_34s = rotl32((c_31s ^ c_26s ^ c_20s ^ c_18s), 1u); \
    const unsigned int c_35s = rotl32((c_32s ^ c_27s ^ c_21s ^ c_19s), 1u); \
    const unsigned int c_36s = rotl32((c_33s ^ c_28s ^ c_22s ^ c_20s), 1u); \
    const unsigned int c_37s = rotl32((c_34s ^ c_29s ^ c_23s ^ c_21s), 1u); \
    const unsigned int c_38s = rotl32((c_35s ^ c_30s ^ c_24s ^ c_22s), 1u); \
    const unsigned int c_39s = rotl32((c_36s ^ c_31s ^ c_25s ^ c_23s), 1u); \
    const unsigned int c_40s = rotl32((c_37s ^ c_32s ^ c_26s ^ c_24s), 1u); \
    const unsigned int c_41s = rotl32((c_38s ^ c_33s ^ c_27s ^ c_25s), 1u); \
    const unsigned int c_42s = rotl32((c_39s ^ c_34s ^ c_28s ^ c_26s), 1u); \
    const unsigned int c_43s = rotl32((c_40s ^ c_35s ^ c_29s ^ c_27s), 1u); \
    const unsigned int c_44s = rotl32((c_41s ^ c_36s ^ c_30s ^ c_28s), 1u); \
    const unsigned int c_45s = rotl32((c_42s ^ c_37s ^ c_31s ^ c_29s), 1u); \
    const unsigned int c_46s = rotl32((c_43s ^ c_38s ^ c_32s ^ c_30s), 1u); \
    const unsigned int c_47s = rotl32((c_44s ^ c_39s ^ c_33s ^ c_31s), 1u); \
    const unsigned int c_48s = rotl32((c_45s ^ c_40s ^ c_34s ^ c_32s), 1u); \
    const unsigned int c_49s = rotl32((c_46s ^ c_41s ^ c_35s ^ c_33s), 1u); \
    const unsigned int c_50s = rotl32((c_47s ^ c_42s ^ c_36s ^ c_34s), 1u); \
    const unsigned int c_51s = rotl32((c_48s ^ c_43s ^ c_37s ^ c_35s), 1u); \
    const unsigned int c_52s = rotl32((c_49s ^ c_44s ^ c_38s ^ c_36s), 1u); \
    const unsigned int c_53s = rotl32((c_50s ^ c_45s ^ c_39s ^ c_37s), 1u); \
    const unsigned int c_54s = rotl32((c_51s ^ c_46s ^ c_40s ^ c_38s), 1u); \
    const unsigned int c_55s = rotl32((c_52s ^ c_47s ^ c_41s ^ c_39s), 1u); \
    const unsigned int c_56s = rotl32((c_53s ^ c_48s ^ c_42s ^ c_40s), 1u); \
    const unsigned int c_57s = rotl32((c_54s ^ c_49s ^ c_43s ^ c_41s), 1u); \
    const unsigned int c_58s = rotl32((c_55s ^ c_50s ^ c_44s ^ c_42s), 1u); \
    const unsigned int c_59s = rotl32((c_56s ^ c_51s ^ c_45s ^ c_43s), 1u); \
    const unsigned int c_60s = rotl32((c_57s ^ c_52s ^ c_46s ^ c_44s), 1u); \
    const unsigned int c_61s = rotl32((c_58s ^ c_53s ^ c_47s ^ c_45s), 1u); \
    const unsigned int c_62s = rotl32((c_59s ^ c_54s ^ c_48s ^ c_46s), 1u); \
    const unsigned int c_63s = rotl32((c_60s ^ c_55s ^ c_49s ^ c_47s), 1u); \
    const unsigned int c_64s = rotl32((c_61s ^ c_56s ^ c_50s ^ c_48s), 1u); \
    const unsigned int c_65s = rotl32((c_62s ^ c_57s ^ c_51s ^ c_49s), 1u); \
    const unsigned int c_66s = rotl32((c_63s ^ c_58s ^ c_52s ^ c_50s), 1u); \
    const unsigned int c_67s = rotl32((c_64s ^ c_59s ^ c_53s ^ c_51s), 1u); \
    const unsigned int c_68s = rotl32((c_65s ^ c_60s ^ c_54s ^ c_52s), 1u); \
    const unsigned int c_69s = rotl32((c_66s ^ c_61s ^ c_55s ^ c_53s), 1u); \
    const unsigned int c_70s = rotl32((c_67s ^ c_62s ^ c_56s ^ c_54s), 1u); \
    const unsigned int c_71s = rotl32((c_68s ^ c_63s ^ c_57s ^ c_55s), 1u); \
    const unsigned int c_72s = rotl32((c_69s ^ c_64s ^ c_58s ^ c_56s), 1u); \
    const unsigned int c_73s = rotl32((c_70s ^ c_65s ^ c_59s ^ c_57s), 1u); \
    const unsigned int c_74s = rotl32((c_71s ^ c_66s ^ c_60s ^ c_58s), 1u); \
    const unsigned int c_75s = rotl32((c_72s ^ c_67s ^ c_61s ^ c_59s), 1u); \
    const unsigned int c_76s = rotl32((c_73s ^ c_68s ^ c_62s ^ c_60s), 1u); \
    const unsigned int c_16sK = c_16s + SHA1C00; \
    const unsigned int c_18sK = c_18s + SHA1C00; \
    const unsigned int c_19sK = c_19s + SHA1C00; \
    const unsigned int c_21sK = c_21s + SHA1C01; \
    const unsigned int c_22sK = c_22s + SHA1C01; \
    const unsigned int c_24sK = c_24s + SHA1C01; \
    const unsigned int c_27sK = c_27s + SHA1C01; \
    const unsigned int c_28sK = c_28s + SHA1C01; \
    const unsigned int c_30sK = c_30s + SHA1C01; \
    const unsigned int c_34sK = c_34s + SHA1C01; \
    const unsigned int c_40sK = c_40s + SHA1C02; \
    const unsigned int c_42sK = c_42s + SHA1C02; \
    const unsigned int c_46sK = c_46s + SHA1C02; \
    const unsigned int c_54sK = c_54s + SHA1C02; \
    const unsigned int c_66sK = c_66s + SHA1C03; \
    const unsigned int c_70sK = c_70s + SHA1C03;

// Precompute the state after rounds 0 and 1. These two rounds have only
// timestamp word w1 as a variable input, and round 0 has none. So we can
// finalize them as compile-time / kernel-time constants plus a single add
// per candidate.
//
// After round 0 (SHA-1 state pos = 5k+1, ignoring base0): variables
//   a = SHA1M_A
//   b = rotl(SHA1M_B, 30)
//   c = SHA1M_C
//   d = SHA1M_D
//   e = T_0 = rotl(SHA1M_A, 5) + ch(SHA1M_B, SHA1M_C, SHA1M_D) + SHA1M_E + SHA1C00 + base0
//
// After round 1: variable a gets rotated to rotl(SHA1M_A, 30), variable d
// gets updated to T_1 = rotl(T_0, 5) + ch(SHA1M_A, rotl(SHA1M_B, 30), SHA1M_C)
//                       + SHA1M_D + SHA1C00 + w1
//                     = R1_OFFSET + w1.
#define DECLARE_HOISTED_R0_R1_CONSTANTS \
    const unsigned int hr_b = rotl32(SHA1M_B, 30u); \
    const unsigned int hr_e = rotl32(SHA1M_A, 5u) + sha1_ch(SHA1M_B, SHA1M_C, SHA1M_D) \
                              + SHA1M_E + SHA1C00 + base0; \
    const unsigned int hr_a = rotl32(SHA1M_A, 30u); \
    const unsigned int hr_d_offset = rotl32(hr_e, 5u) \
                                     + sha1_ch(SHA1M_A, hr_b, SHA1M_C) \
                                     + SHA1M_D + SHA1C00;

#define RUN_POLL_GROUP(timestamp0) do { \
    RUN_CANDIDATE(timestamp0); \
    RUN_CANDIDATE((timestamp0) + stride); \
    RUN_CANDIDATE((timestamp0) + stride2); \
    RUN_CANDIDATE((timestamp0) + stride3); \
} while (0)

// Run one SHA-1 candidate against the prefix. The message word `w1` is the
// supplied timestamp; the other 15 words are the precomputed `base*` locals.
// The expansion uses precomputed `c_NN(s|sK)` from DECLARE_EXPANSION_CONSTANTS.
//
// Skips rounds 0 and 1 by starting from the post-round-1 state (constants
// captured in hr_a, hr_b, hr_e plus hr_d_offset + w1).
//
// Skips rounds 77, 78, 79: the final D_80 and E_80 outputs are
// rotl(T_76, 30) and rotl(T_75, 30), recovered by rotating the live
// variables d and e by 30 after round 76's STEP.
//
// The w1-contribution table per round is the linear propagation of w1 through
// the SHA-1 expansion (mirror of hashcat's w0 trick in m00100_a3-optimized,
// shifted by 1 round because our variable word is at index 1).
#define RUN_CANDIDATE(timestamp) do { \
    const unsigned int w1     = (timestamp); \
    const unsigned int w1s01  = rotl32(w1, 1u); \
    const unsigned int w1s02  = rotl32(w1, 2u); \
    const unsigned int w1s03  = rotl32(w1, 3u); \
    const unsigned int w1s04  = rotl32(w1, 4u); \
    const unsigned int w1s05  = rotl32(w1, 5u); \
    const unsigned int w1s06  = rotl32(w1, 6u); \
    const unsigned int w1s07  = rotl32(w1, 7u); \
    const unsigned int w1s08  = rotl32(w1, 8u); \
    const unsigned int w1s09  = rotl32(w1, 9u); \
    const unsigned int w1s10  = rotl32(w1, 10u); \
    const unsigned int w1s11  = rotl32(w1, 11u); \
    const unsigned int w1s12  = rotl32(w1, 12u); \
    const unsigned int w1s13  = rotl32(w1, 13u); \
    const unsigned int w1s14  = rotl32(w1, 14u); \
    const unsigned int w1s15  = rotl32(w1, 15u); \
    const unsigned int w1s16  = rotl32(w1, 16u); \
    const unsigned int w1s17  = rotl32(w1, 17u); \
    const unsigned int w1s18  = rotl32(w1, 18u); \
    const unsigned int w1s19  = rotl32(w1, 19u); \
    const unsigned int w1s20  = rotl32(w1, 20u); \
    const unsigned int w1s04_06     = w1s04 ^ w1s06; \
    const unsigned int w1s04_08     = w1s04 ^ w1s08; \
    const unsigned int w1s08_12     = w1s08 ^ w1s12; \
    const unsigned int w1s04_06_07  = w1s04_06 ^ w1s07; \
    unsigned int a = hr_a; \
    unsigned int b = hr_b; \
    unsigned int c = SHA1M_C; \
    unsigned int d = hr_d_offset + w1; \
    unsigned int e = hr_e; \
    /* Rounds 0..1 are hoisted; resume from round 2. */ \
    /* Round 2..15: F0, K = SHA1C00. */ \
    SHA1_STEP(SHA1_F0, SHA1C00, d, e, a, b, c, base2); \
    SHA1_STEP(SHA1_F0, SHA1C00, c, d, e, a, b, base3); \
    SHA1_STEP(SHA1_F0, SHA1C00, b, c, d, e, a, base4); \
    SHA1_STEP(SHA1_F0, SHA1C00, a, b, c, d, e, base5); \
    SHA1_STEP(SHA1_F0, SHA1C00, e, a, b, c, d, base6); \
    SHA1_STEP(SHA1_F0, SHA1C00, d, e, a, b, c, base7); \
    SHA1_STEP(SHA1_F0, SHA1C00, c, d, e, a, b, base8); \
    SHA1_STEP(SHA1_F0, SHA1C00, b, c, d, e, a, base9); \
    SHA1_STEP(SHA1_F0, SHA1C00, a, b, c, d, e, basea); \
    SHA1_STEP(SHA1_F0, SHA1C00, e, a, b, c, d, baseb); \
    SHA1_STEP(SHA1_F0, SHA1C00, d, e, a, b, c, basec); \
    SHA1_STEP(SHA1_F0, SHA1C00, c, d, e, a, b, based); \
    SHA1_STEP(SHA1_F0, SHA1C00, b, c, d, e, a, basee); \
    SHA1_STEP(SHA1_F0, SHA1C00, a, b, c, d, e, basef); \
    /* Rounds 16..19: F0, K = SHA1C00. Only round 17 sees w1 (as w1s01). */ \
    SHA1_STEP_K(SHA1_F0, e, a, b, c, d, c_16sK); \
    SHA1_STEP  (SHA1_F0, SHA1C00, d, e, a, b, c, (c_17s ^ w1s01)); \
    SHA1_STEP_K(SHA1_F0, c, d, e, a, b, c_18sK); \
    SHA1_STEP_K(SHA1_F0, b, c, d, e, a, c_19sK); \
    /* Rounds 20..39: F1, K = SHA1C01. */ \
    SHA1_STEP  (SHA1_F1, SHA1C01, a, b, c, d, e, (c_20s ^ w1s02)); \
    SHA1_STEP_K(SHA1_F1, e, a, b, c, d, c_21sK); \
    SHA1_STEP_K(SHA1_F1, d, e, a, b, c, c_22sK); \
    SHA1_STEP  (SHA1_F1, SHA1C01, c, d, e, a, b, (c_23s ^ w1s03)); \
    SHA1_STEP_K(SHA1_F1, b, c, d, e, a, c_24sK); \
    SHA1_STEP  (SHA1_F1, SHA1C01, a, b, c, d, e, (c_25s ^ w1s02)); \
    SHA1_STEP  (SHA1_F1, SHA1C01, e, a, b, c, d, (c_26s ^ w1s04)); \
    SHA1_STEP_K(SHA1_F1, d, e, a, b, c, c_27sK); \
    SHA1_STEP_K(SHA1_F1, c, d, e, a, b, c_28sK); \
    SHA1_STEP  (SHA1_F1, SHA1C01, b, c, d, e, a, (c_29s ^ w1s05)); \
    SHA1_STEP_K(SHA1_F1, a, b, c, d, e, c_30sK); \
    SHA1_STEP  (SHA1_F1, SHA1C01, e, a, b, c, d, (c_31s ^ w1s02 ^ w1s04)); \
    SHA1_STEP  (SHA1_F1, SHA1C01, d, e, a, b, c, (c_32s ^ w1s06)); \
    SHA1_STEP  (SHA1_F1, SHA1C01, c, d, e, a, b, (c_33s ^ w1s02 ^ w1s03)); \
    SHA1_STEP_K(SHA1_F1, b, c, d, e, a, c_34sK); \
    SHA1_STEP  (SHA1_F1, SHA1C01, a, b, c, d, e, (c_35s ^ w1s07)); \
    SHA1_STEP  (SHA1_F1, SHA1C01, e, a, b, c, d, (c_36s ^ w1s04)); \
    SHA1_STEP  (SHA1_F1, SHA1C01, d, e, a, b, c, (c_37s ^ w1s04_06)); \
    SHA1_STEP  (SHA1_F1, SHA1C01, c, d, e, a, b, (c_38s ^ w1s08)); \
    SHA1_STEP  (SHA1_F1, SHA1C01, b, c, d, e, a, (c_39s ^ w1s04)); \
    /* Rounds 40..59: F2, K = SHA1C02. */ \
    SHA1_STEP_K(SHA1_F2, a, b, c, d, e, c_40sK); \
    SHA1_STEP  (SHA1_F2, SHA1C02, e, a, b, c, d, (c_41s ^ w1s04 ^ w1s09)); \
    SHA1_STEP_K(SHA1_F2, d, e, a, b, c, c_42sK); \
    SHA1_STEP  (SHA1_F2, SHA1C02, c, d, e, a, b, (c_43s ^ w1s06 ^ w1s08)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, b, c, d, e, a, (c_44s ^ w1s10)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, a, b, c, d, e, (c_45s ^ w1s03 ^ w1s06 ^ w1s07)); \
    SHA1_STEP_K(SHA1_F2, e, a, b, c, d, c_46sK); \
    SHA1_STEP  (SHA1_F2, SHA1C02, d, e, a, b, c, (c_47s ^ w1s04 ^ w1s11)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, c, d, e, a, b, (c_48s ^ w1s04_08)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, b, c, d, e, a, (c_49s ^ w1s03 ^ w1s04_08 ^ w1s05 ^ w1s10)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, a, b, c, d, e, (c_50s ^ w1s12)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, e, a, b, c, d, (c_51s ^ w1s08)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, d, e, a, b, c, (c_52s ^ w1s04_06)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, c, d, e, a, b, (c_53s ^ w1s04_08 ^ w1s13)); \
    SHA1_STEP_K(SHA1_F2, b, c, d, e, a, c_54sK); \
    SHA1_STEP  (SHA1_F2, SHA1C02, a, b, c, d, e, (c_55s ^ w1s07 ^ w1s10 ^ w1s12)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, e, a, b, c, d, (c_56s ^ w1s14)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, d, e, a, b, c, (c_57s ^ w1s04_06_07 ^ w1s10 ^ w1s11)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, c, d, e, a, b, (c_58s ^ w1s08)); \
    SHA1_STEP  (SHA1_F2, SHA1C02, b, c, d, e, a, (c_59s ^ w1s04_08 ^ w1s15)); \
    /* Rounds 60..76: F1, K = SHA1C03. Rounds 77..79 are skipped. */ \
    SHA1_STEP  (SHA1_F1, SHA1C03, a, b, c, d, e, (c_60s ^ w1s08_12)); \
    SHA1_STEP  (SHA1_F1, SHA1C03, e, a, b, c, d, (c_61s ^ w1s04 ^ w1s07 ^ w1s08_12 ^ w1s14)); \
    SHA1_STEP  (SHA1_F1, SHA1C03, d, e, a, b, c, (c_62s ^ w1s16)); \
    SHA1_STEP  (SHA1_F1, SHA1C03, c, d, e, a, b, (c_63s ^ w1s04_06 ^ w1s08_12)); \
    SHA1_STEP  (SHA1_F1, SHA1C03, b, c, d, e, a, (c_64s ^ w1s08)); \
    SHA1_STEP  (SHA1_F1, SHA1C03, a, b, c, d, e, (c_65s ^ w1s04_06_07 ^ w1s08_12 ^ w1s17)); \
    SHA1_STEP_K(SHA1_F1, e, a, b, c, d, c_66sK); \
    SHA1_STEP  (SHA1_F1, SHA1C03, d, e, a, b, c, (c_67s ^ w1s14 ^ w1s16)); \
    SHA1_STEP  (SHA1_F1, SHA1C03, c, d, e, a, b, (c_68s ^ w1s08 ^ w1s18)); \
    SHA1_STEP  (SHA1_F1, SHA1C03, b, c, d, e, a, (c_69s ^ w1s11 ^ w1s14 ^ w1s15)); \
    SHA1_STEP_K(SHA1_F1, a, b, c, d, e, c_70sK); \
    SHA1_STEP  (SHA1_F1, SHA1C03, e, a, b, c, d, (c_71s ^ w1s12 ^ w1s19)); \
    SHA1_STEP  (SHA1_F1, SHA1C03, d, e, a, b, c, (c_72s ^ w1s12 ^ w1s16)); \
    SHA1_STEP  (SHA1_F1, SHA1C03, c, d, e, a, b, (c_73s ^ w1s05 ^ w1s11 ^ w1s12 ^ w1s13 ^ w1s16 ^ w1s18)); \
    SHA1_STEP  (SHA1_F1, SHA1C03, b, c, d, e, a, (c_74s ^ w1s20)); \
    /* Round 75 needs the full SHA1_STEP: round 76's macro c-arg is variable b,
       which holds C_76 = rotl(B_75, 30) — that rotation is produced inside
       round 75's b-update. */ \
    SHA1_STEP(SHA1_F1, SHA1C03, a, b, c, d, e, (c_75s ^ w1s08 ^ w1s16)); \
    /* Round 76: produces T_76 in variable d. Variable a (macro b-arg) gets
       rotated to C_77 which we don't need afterwards — use STEP_FINAL to
       skip that dead rotation. */ \
    SHA1_STEP_FINAL(SHA1_F1, SHA1C03, e, a, b, c, d, (c_76s ^ w1s06 ^ w1s12 ^ w1s14)); \
    /* Skip rounds 77, 78, 79. Recover D_80 = rotl(T_76, 30),
       E_80 = rotl(T_75, 30) directly from variables d and e. */ \
    const unsigned int key_id_high = SHA1M_D + rotl32(d, 30u); \
    const unsigned int key_id_low  = SHA1M_E + rotl32(e, 30u); \
    for (unsigned int prefix_index = 0u; prefix_index < prefix_count; ++prefix_index) { \
        const unsigned int prefix_mask_high  = prefixes[prefix_index * 4u + 0u]; \
        const unsigned int prefix_mask_low   = prefixes[prefix_index * 4u + 1u]; \
        const unsigned int prefix_value_high = prefixes[prefix_index * 4u + 2u]; \
        const unsigned int prefix_value_low  = prefixes[prefix_index * 4u + 3u]; \
        const unsigned int diff_high = ((key_id_high) ^ prefix_value_high) & prefix_mask_high; \
        const unsigned int diff_low  = ((key_id_low)  ^ prefix_value_low)  & prefix_mask_low; \
        if (max_error == 0u) { \
            if ((diff_high | diff_low) == 0u) { \
                atomicMin((unsigned int*) best_timestamp, (timestamp)); \
                return; \
            } \
        } else { \
            /* Collapse each nibble to its low bit (set iff nonzero), then count
               the set bits to get the number of wrong hex digits. */ \
            const unsigned int nz_high = (diff_high | (diff_high >> 1u) | (diff_high >> 2u) \
                                          | (diff_high >> 3u)) & 0x11111111u; \
            const unsigned int nz_low  = (diff_low  | (diff_low  >> 1u) | (diff_low  >> 2u) \
                                          | (diff_low  >> 3u)) & 0x11111111u; \
            if ((unsigned int) (__popc(nz_high) + __popc(nz_low)) <= max_error) { \
                atomicMin((unsigned int*) best_timestamp, (timestamp)); \
                return; \
            } \
        } \
    } \
} while (0)

extern "C" __global__ __launch_bounds__(FIXED_BLOCK_SIZE)
void search_v4_ed25519(
    const unsigned int* __restrict__ base_words,
    const unsigned int start_timestamp,
    const unsigned int count,
    const unsigned int* __restrict__ prefixes,
    const unsigned int prefix_count,
    const unsigned int max_error,
    volatile unsigned int* __restrict__ best_timestamp)
{
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int stride2 = stride * 2u;
    const unsigned int stride3 = stride * 3u;
    const unsigned int poll_stride = stride * POLL_INTERVAL;

    const unsigned int base0 = base_words[0];
    const unsigned int base2 = base_words[2];
    const unsigned int base3 = base_words[3];
    const unsigned int base4 = base_words[4];
    const unsigned int base5 = base_words[5];
    const unsigned int base6 = base_words[6];
    const unsigned int base7 = base_words[7];
    const unsigned int base8 = base_words[8];
    const unsigned int base9 = base_words[9];
    const unsigned int basea = base_words[10];
    const unsigned int baseb = base_words[11];
    const unsigned int basec = base_words[12];
    const unsigned int based = base_words[13];
    const unsigned int basee = base_words[14];
    const unsigned int basef = base_words[15];

    DECLARE_EXPANSION_CONSTANTS
    DECLARE_HOISTED_R0_R1_CONSTANTS

    for (unsigned int index = gid; index < count;) {
        const unsigned int best_so_far = load_best_timestamp(best_timestamp);
        const unsigned int first_timestamp = start_timestamp + index;

        if ((best_so_far != NOT_FOUND_TIMESTAMP) && (best_so_far <= first_timestamp)) {
            return;
        }

        for (unsigned int group = 0; group < MATCH_POLL_GROUPS && index < count; ++group) {
            const unsigned int timestamp0 = start_timestamp + index;
            const unsigned int remaining = count - index;

            RUN_CANDIDATE(timestamp0);

            if (remaining > stride) {
                RUN_CANDIDATE(timestamp0 + stride);
            }

            if (remaining > stride2) {
                RUN_CANDIDATE(timestamp0 + stride2);
            }

            if (remaining > stride3) {
                RUN_CANDIDATE(timestamp0 + stride3);
            }

            if (remaining <= poll_stride) {
                return;
            }

            index += poll_stride;
        }
    }
}

extern "C" __global__ __launch_bounds__(FIXED_BLOCK_SIZE)
void search_v4_ed25519_full(
    const unsigned int* __restrict__ base_words,
    const unsigned int start_timestamp,
    const unsigned int* __restrict__ prefixes,
    const unsigned int prefix_count,
    const unsigned int max_error,
    volatile unsigned int* __restrict__ best_timestamp)
{
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int stride2 = stride * 2u;
    const unsigned int stride3 = stride * 3u;
    const unsigned int poll_stride = stride * POLL_INTERVAL;

    const unsigned int base0 = base_words[0];
    const unsigned int base2 = base_words[2];
    const unsigned int base3 = base_words[3];
    const unsigned int base4 = base_words[4];
    const unsigned int base5 = base_words[5];
    const unsigned int base6 = base_words[6];
    const unsigned int base7 = base_words[7];
    const unsigned int base8 = base_words[8];
    const unsigned int base9 = base_words[9];
    const unsigned int basea = base_words[10];
    const unsigned int baseb = base_words[11];
    const unsigned int basec = base_words[12];
    const unsigned int based = base_words[13];
    const unsigned int basee = base_words[14];
    const unsigned int basef = base_words[15];

    DECLARE_EXPANSION_CONSTANTS
    DECLARE_HOISTED_R0_R1_CONSTANTS

    unsigned int timestamp0 = start_timestamp + gid;

    for (unsigned int iteration = 0; iteration < FULL_BATCH_OUTER_LOOPS;) {
        const unsigned int best_so_far = load_best_timestamp(best_timestamp);

        if ((best_so_far != NOT_FOUND_TIMESTAMP) && (best_so_far <= timestamp0)) {
            return;
        }

        for (unsigned int group = 0;
             group < MATCH_POLL_GROUPS && iteration < FULL_BATCH_OUTER_LOOPS;
             ++group, ++iteration) {
            RUN_POLL_GROUP(timestamp0);
            timestamp0 += poll_stride;
        }
    }
}

#undef RUN_POLL_GROUP
#undef RUN_CANDIDATE
#undef DECLARE_HOISTED_R0_R1_CONSTANTS
#undef DECLARE_EXPANSION_CONSTANTS
#undef SHA1_STEP_FINAL
#undef SHA1_STEP_K
#undef SHA1_STEP
#undef SHA1_F2
#undef SHA1_F1
#undef SHA1_F0
#undef MATCH_POLL_GROUPS
#undef FULL_BATCH_OUTER_LOOPS
#undef POLL_INTERVAL
#undef NOT_FOUND_TIMESTAMP
#undef SHA1M_E
#undef SHA1M_D
#undef SHA1M_C
#undef SHA1M_B
#undef SHA1M_A
#undef SHA1C03
#undef SHA1C02
#undef SHA1C01
#undef SHA1C00
