mod fingerprint;
mod gpu;
mod output;
mod search;

pub use fingerprint::{
    HexPrefix, HexPrefixSet, MAX_PREFIXES, fingerprint_v4_ed25519, hex_upper,
    key_id_from_fingerprint, key_id_hex, seed_from_hex,
};
pub use gpu::gpu_search_available;
pub use output::{build_armored_secret_key_block, encrypt_for_recipient};
pub use search::{SearchConfig, SearchResult, search};

pub fn format_uid(name: &str, email: &str) -> String {
    format!("{name} <{email}>")
}
