use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use ed25519_dalek::SigningKey;
use pgp_vanity::{
    build_armored_secret_key_block, fingerprint_v4_ed25519, format_uid, hex_upper,
    key_id_from_fingerprint, key_id_hex,
};
use tempfile::TempDir;

#[test]
fn fingerprint_formula_matches_gpg_export() -> Result<()> {
    if !gpg_available() {
        return Ok(());
    }

    let gpg_home = GpgHome::new()?;
    let uid = format_uid("Verifier User", "verify@example.com");

    let mut generate = gpg(gpg_home.path());
    generate
        .arg("--batch")
        .arg("--pinentry-mode")
        .arg("loopback")
        .arg("--passphrase")
        .arg("")
        .arg("--quick-generate-key")
        .arg(&uid)
        .arg("ed25519")
        .arg("sign")
        .arg("0");
    run_command(generate)?;

    let public_key_path = gpg_home.path().join("pubkey.bin");
    let mut export = gpg(gpg_home.path());
    export
        .arg("--output")
        .arg(&public_key_path)
        .arg("--export")
        .arg(&uid);
    run_command(export)?;

    let mut list = gpg(gpg_home.path());
    list.arg("--list-secret-keys")
        .arg("--with-colons")
        .arg("--with-fingerprint")
        .arg(&uid);
    let listing = run_command(list)?;

    let expected_key_id = colon_field(first_line_with_prefix(&listing, "sec:")?, 4)?;
    let expected_fingerprint = colon_field(first_line_with_prefix(&listing, "fpr:")?, 9)?;

    let packet = fs::read(public_key_path)?;
    assert_eq!(&packet[..2], &[0x98, 0x33]);
    assert_eq!(packet[2], 0x04);

    let timestamp = u32::from_be_bytes(packet[3..7].try_into().expect("timestamp slice"));
    let public_key: [u8; 32] = packet[21..53].try_into().expect("public key slice");

    let fingerprint = fingerprint_v4_ed25519(&public_key, timestamp);
    assert_eq!(hex_upper(fingerprint), expected_fingerprint);
    assert_eq!(
        key_id_hex(key_id_from_fingerprint(&fingerprint)),
        expected_key_id
    );

    Ok(())
}

#[test]
fn generated_private_key_block_imports_into_gpg() -> Result<()> {
    if !gpg_available() {
        return Ok(());
    }

    let seed = [0x42; 32];
    let timestamp = 0x1234_5678;
    let uid = format_uid("Vanity Test", "vanity@example.com");
    let armored = build_armored_secret_key_block(&seed, timestamp, &uid)?;

    assert!(armored.starts_with("-----BEGIN PGP PRIVATE KEY BLOCK-----"));
    assert!(!armored.contains("\nComment:"));
    assert!(armored.contains("-----END PGP PRIVATE KEY BLOCK-----"));

    let gpg_home = GpgHome::new()?;
    let block_path = gpg_home.path().join("vanity.asc");
    fs::write(&block_path, armored.as_bytes())?;

    let mut import = gpg(gpg_home.path());
    import.arg("--batch").arg("--import").arg(&block_path);
    run_command(import)?;

    let mut list = gpg(gpg_home.path());
    list.arg("--list-secret-keys")
        .arg("--with-colons")
        .arg("--with-fingerprint");
    let listing = run_command(list)?;

    let signing_key = SigningKey::from_bytes(&seed);
    let public_key = signing_key.verifying_key().to_bytes();
    let fingerprint = fingerprint_v4_ed25519(&public_key, timestamp);
    let expected_fingerprint = hex_upper(fingerprint);
    let expected_key_id = key_id_hex(key_id_from_fingerprint(&fingerprint));

    assert_eq!(
        colon_field(first_line_with_prefix(&listing, "sec:")?, 4)?,
        expected_key_id
    );
    assert_eq!(
        colon_field(first_line_with_prefix(&listing, "fpr:")?, 9)?,
        expected_fingerprint
    );
    assert!(listing.contains(&uid));

    Ok(())
}

struct GpgHome {
    temp_dir: TempDir,
}

impl GpgHome {
    fn new() -> Result<Self> {
        Ok(Self {
            temp_dir: TempDir::new().context("failed to create temporary GnuPG home")?,
        })
    }

    fn path(&self) -> &Path {
        self.temp_dir.path()
    }
}

impl Drop for GpgHome {
    fn drop(&mut self) {
        let _ = Command::new("gpgconf")
            .arg("--homedir")
            .arg(self.path())
            .arg("--kill")
            .arg("all")
            .status();
        thread::sleep(Duration::from_millis(50));
    }
}

fn gpg(home: &Path) -> Command {
    let mut command = Command::new("gpg");
    command.arg("--homedir").arg(home);
    command
}

fn gpg_available() -> bool {
    Command::new("gpg")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn run_command(mut command: Command) -> Result<String> {
    let description = describe_command(&command);
    let output = command
        .output()
        .with_context(|| format!("failed to run {description}"))?;
    ensure_success(description, output)
}

fn ensure_success(description: String, output: Output) -> Result<String> {
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }

    bail!(
        "command failed: {description}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn describe_command(command: &Command) -> String {
    let program = command.get_program().to_string_lossy();
    let args = command
        .get_args()
        .map(os_str_to_string)
        .collect::<Vec<_>>()
        .join(" ");
    format!("{program} {args}")
}

fn os_str_to_string(value: &OsStr) -> String {
    value.to_string_lossy().into_owned()
}

fn first_line_with_prefix<'a>(text: &'a str, prefix: &str) -> Result<&'a str> {
    text.lines()
        .find(|line| line.starts_with(prefix))
        .with_context(|| format!("missing {prefix} line in:\n{text}"))
}

fn colon_field(line: &str, index: usize) -> Result<String> {
    line.split(':')
        .nth(index)
        .map(ToOwned::to_owned)
        .with_context(|| format!("missing colon field {index} in line: {line}"))
}
