use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use pgp_vanity::{
    HexPrefixSet, SearchConfig, build_armored_secret_key_block, encrypt_for_recipient, format_uid,
    search, seed_from_hex,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Generate a vanity GnuPG-compatible v4 Ed25519 private key block using a GPU (HIP or CUDA) search over creation timestamps."
)]
struct Cli {
    #[arg(
        required = true,
        num_args = 1..,
        help = "One or more hex prefixes to match against the 16-hex-character primary key ID; the search returns as soon as a key matches any of them"
    )]
    prefixes: Vec<String>,

    #[arg(long, help = "Real name for the OpenPGP User ID")]
    name: String,

    #[arg(long, help = "Email address for the OpenPGP User ID")]
    email: String,

    #[arg(long, help = "Disable progress updates on stderr")]
    no_progress: bool,

    #[arg(
        long,
        value_name = "HEX",
        help = "Use this 32-byte Ed25519 seed instead of generating random keys"
    )]
    seed_hex: Option<String>,

    #[arg(
        long,
        value_name = "N",
        help = "Stop after scanning N random keys without a match"
    )]
    max_key_attempts: Option<u64>,

    #[arg(
        long,
        value_name = "N",
        default_value_t = 0,
        help = "Allow up to N wrong hex digits within a prefix; e.g. --max-error 1 AAAA matches AAA0 and ABAA"
    )]
    max_error: u32,

    #[arg(
        long,
        value_name = "FILE",
        help = "Encrypt the armored secret key block to this recipient's OpenPGP public key before writing to stdout"
    )]
    encrypt_to: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let prefixes = HexPrefixSet::parse(&cli.prefixes)?.with_max_error(cli.max_error);
    let fixed_seed = cli.seed_hex.as_deref().map(seed_from_hex).transpose()?;

    let prefix_display = prefixes.normalized_list().join(", ");
    if cli.max_error > 0 {
        eprintln!(
            "searching for key ID prefix {prefix_display} (up to {} wrong hex digit(s))",
            cli.max_error
        );
    } else {
        eprintln!("searching for key ID prefix {prefix_display}");
    }

    let result = search(SearchConfig {
        prefixes,
        fixed_seed,
        max_key_attempts: cli.max_key_attempts,
        progress: !cli.no_progress,
    })?;

    eprintln!("found matching key ID {}", result.key_id_hex());
    eprintln!("fingerprint: {}", result.fingerprint_hex());
    eprintln!("timestamp: {}", result.timestamp);
    if cli.encrypt_to.is_none() {
        eprintln!("seed: {}", result.seed_hex());
    } else {
        eprintln!("seed: <redacted; included in the encrypted output>");
    }
    eprintln!("public key: {}", result.public_key_hex());
    eprintln!("keys tried: {}", result.key_attempts);
    eprintln!("timestamps checked: {}", result.timestamps_checked);
    eprintln!("elapsed: {:.2}s", result.elapsed.as_secs_f64());

    let uid = format_uid(&cli.name, &cli.email);
    let armored = build_armored_secret_key_block(&result.seed, result.timestamp, &uid)?;

    let output = match &cli.encrypt_to {
        Some(path) => {
            let recipient = fs::read(path)
                .with_context(|| format!("failed to read recipient public key from {path:?}"))?;
            encrypt_for_recipient(&armored, &recipient)?
        }
        None => armored,
    };

    print!("{output}");
    Ok(())
}
