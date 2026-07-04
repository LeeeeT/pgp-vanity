use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use pgp_vanity::gpu_search_available;

const FIXED_SEED_HEX: &str = "265D84356D04A0D0D74E7A5CB270DE6C6D3100F264F93338C5804568BECFF5FC";

#[test]
fn progress_enabled_cli_finishes_and_emits_private_key_block() -> Result<()> {
    if !gpu_search_available() {
        return Ok(());
    }

    let binary = env!("CARGO_BIN_EXE_pgp-vanity");
    let mut child = Command::new(binary)
        .arg("1")
        .arg("--name")
        .arg("Alice")
        .arg("--email")
        .arg("alice@example.com")
        .arg("--seed-hex")
        .arg(FIXED_SEED_HEX)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn pgp-vanity binary")?;

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait()? {
            let mut stdout = String::new();
            let mut stderr = String::new();

            child
                .stdout
                .take()
                .expect("stdout pipe missing")
                .read_to_string(&mut stdout)?;
            child
                .stderr
                .take()
                .expect("stderr pipe missing")
                .read_to_string(&mut stderr)?;

            if !status.success() {
                bail!("binary exited with {status}\nstdout:\n{stdout}\nstderr:\n{stderr}");
            }

            assert!(stdout.starts_with("-----BEGIN PGP PRIVATE KEY BLOCK-----"));
            assert!(stdout.contains("-----END PGP PRIVATE KEY BLOCK-----"));
            assert!(stderr.contains("found matching key ID"));
            return Ok(());
        }

        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();

            let mut stdout = String::new();
            let mut stderr = String::new();
            if let Some(mut handle) = child.stdout.take() {
                let _ = handle.read_to_string(&mut stdout);
            }
            if let Some(mut handle) = child.stderr.take() {
                let _ = handle.read_to_string(&mut stderr);
            }

            bail!("binary did not finish within the timeout\nstdout:\n{stdout}\nstderr:\n{stderr}");
        }

        thread::sleep(Duration::from_millis(20));
    }
}
