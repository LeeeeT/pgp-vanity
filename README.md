# pgp-vanity

`pgp-vanity` is a GPU-accelerated CLI for generating vanity OpenPGP private keys whose primary key ID starts with a chosen hexadecimal prefix.

It uses a GPU search kernel, compiled at runtime for either AMD (HIP via HIPRTC) or NVIDIA (CUDA via NVRTC) GPUs, to scan the 32-bit creation timestamp space for a fixed Ed25519 keypair, then emits a complete ASCII-armored `PGP PRIVATE KEY BLOCK` when it finds a match.

```console
$ pgp-vanity --name Alice --email alice@example.com --max-error 2 AAAAAAAAAAAAAAAA
searching for key ID prefix AAAAAAAAAAAAAAAA (up to 2 wrong hex digit(s))
using HIP GPU: AMD Radeon RX 6900 XT
found matching key ID AAAAEAAAAAAAAAAF
fingerprint: C12646C6CA7CBD822750D50EAAAAEAAAAAAAAAAF
timestamp: 1944069932
seed: 07606D6E2582A1AFE7142D2F001527F1E4829667E7C287800886A886BBA0487C
public key: 9ACEE20A6AF7B86F22D0DF0EE83D79D00697DF5EC27D9E78626360576FFAAED1
keys tried: 1189
timestamps checked: 5104365217581
elapsed: 248.85s
-----BEGIN PGP PRIVATE KEY BLOCK-----
```

(ran at 20 GH/s)

## Compatibility

The current output format is intentionally conservative and broadly practical:

- OpenPGP v4 primary keys
- Ed25519 encoded as legacy EdDSA
- Unencrypted secret-key material (unless `--encrypt-to` is used)

That format is chosen because it imports cleanly into current GnuPG releases, while RFC 9580 v6 support is still not broadly deployed.

## Requirements

- Rust and Cargo to build from source
- a supported GPU with its runtime installed:
  - **AMD**: a GPU supported by ROCm and a working ROCm/HIP runtime (`libamdhip64.so` and `libhiprtc.so`)
  - **NVIDIA**: a working NVIDIA driver (`libcuda.so` / `nvcuda.dll`) and the CUDA toolkit for NVRTC (`libnvrtc.so` / `nvrtc64_*.dll`)

There is no CPU fallback; a GPU is required.

The backend is selected automatically at startup: HIP is tried first, then CUDA. Set `PGP_VANITY_GPU_BACKEND=hip` or `PGP_VANITY_GPU_BACKEND=cuda` to force a specific backend.

Note for NVIDIA users: CUDA 13 dropped support for Maxwell, Pascal, and Volta. If you have an older GPU (compute capability < 7.5), install an older CUDA toolkit that still targets your architecture.

## Quick Start

Build and run with Cargo:

```bash
cargo run --release -- --help
cargo run --release -- 1ABC --name "Alice Example" --email alice@example.com
```

With Nix, the flake wrappers set up the GPU runtime library paths automatically:

```bash
# AMD (HIP):
nix run .#hip -- --name Alice --email alice@example.com A

# NVIDIA (CUDA):
nix run .#cuda -- --name Alice --email alice@example.com A
```

The `.#cuda` variant bundles NVRTC from nixpkgs (an unfree package, allowed for just this output by the flake) and loads `libcuda.so.1` from the host NVIDIA driver.

If you want a development shell with `cargo`, `rustc`, `clippy`, and ROCm tools available:

```bash
nix develop
```

## Features

- GPU search over the full 32-bit creation timestamp range, on HIP or CUDA
- Multiple prefixes searched simultaneously, with optional error tolerance (`--max-error`)
- Minimal key format that imports cleanly into GnuPG
- Optional encryption of the result to a recipient's OpenPGP key (`--encrypt-to`)
- Progress written to stderr so stdout remains safe to pipe into a file
- Deterministic reproduction with `--seed-hex`
- Per-device kernel auto-tuning, cached between runs
- Integration tests covering fingerprint derivation, GnuPG import, and CLI behavior

## Usage

```text
pgp-vanity [OPTIONS] --name <NAME> --email <EMAIL> <PREFIX>...
```

Arguments:

- `<PREFIX>...`: one or more hexadecimal prefixes to match against the 16-hex-character primary key ID; the search returns as soon as a key matches any of them

Options:

- `--name <NAME>`: real name for the OpenPGP User ID
- `--email <EMAIL>`: email address for the OpenPGP User ID
- `--no-progress`: disable live progress updates on stderr
- `--seed-hex <HEX>`: reuse a specific 32-byte Ed25519 seed instead of generating random keys
- `--max-key-attempts <N>`: stop after scanning `N` random keys without a match
- `--max-error <N>`: allow up to `N` wrong hex digits within a prefix; e.g. `--max-error 1 AAAA` matches `AAA0` and `ABAA`
- `--encrypt-to <FILE>`: encrypt the armored secret key block to this recipient's OpenPGP public key before writing to stdout (and redact the seed from stderr)

## Examples

Search for a one-nybble prefix:

```bash
pgp-vanity 1 --name Alice --email alice@example.com
```

Search for any of several prefixes at once (matches the first one found):

```bash
pgp-vanity DEAD BEEF C0FFEE --name Alice --email alice@example.com
```

Search for a longer prefix and write the armored key block to a file:

```bash
pgp-vanity AAAA --name Alice --email alice@example.com > vanity-private.asc
```

Reproduce a search with a fixed seed:

```bash
pgp-vanity AAAA \
  --name Alice \
  --email alice@example.com \
  --seed-hex 265D84356D04A0D0D74E7A5CB270DE6C6D3100F264F93338C5804568BECFF5FC \
  --max-key-attempts 1
```

Import the generated key into GnuPG:

```bash
gpg --import vanity-private.asc
```

The search draws creation timestamps from the entire 32-bit range, so a matching key is often dated in the future. GnuPG refuses to use such a key until its creation time has passed ("key was created N days in the future"), so pass `--faked-system-time` set to the key's creation time (the `timestamp` line on stderr) or later:

```bash
gpg --faked-system-time 1944069932! --import vanity-private.asc
```

The trailing `!` freezes the clock at that instant instead of letting it keep ticking. Any later operation that uses the key (signing, certifying) needs the same option; add `faked-system-time` to `gpg.conf` to apply it permanently.

Encrypt the result to your own OpenPGP key so the private material never appears as plaintext on disk or the terminal:

```bash
gpg --export --armor you@example.com > recipient.asc
pgp-vanity AAAA --name Alice --email alice@example.com \
  --encrypt-to recipient.asc > vanity-private.asc.pgp
gpg --decrypt vanity-private.asc.pgp | gpg --import
```

## Progress Output

Progress is written to stderr and updated in place on a single line.

The live status line includes:

- overall explored prefix space, based on `16^prefix_len`
- current throughput in GH/s
- an ETA estimate for the remaining search space

For example, a prefix of `AAAA` corresponds to `65,536` high-order key-ID combinations.

The CLI also prints the GPU backend and device it selected before the search starts. Depending on the runtime, the device name may be a marketing name or an architecture-style identifier such as `gfx1030`.

## How It Works

For GnuPG-compatible v4 Ed25519 keys, the primary key fingerprint is:

```text
SHA1(0x99 || uint16_be(len(public_key_packet_body)) || public_key_packet_body)
```

The key ID is the low 64 bits of that fingerprint.

For each keypair, `pgp-vanity` keeps the Ed25519 secret/public key fixed and varies the 32-bit creation timestamp on the GPU. That is much cheaper than regenerating a brand-new keypair for every attempt, and the timestamp still affects the fingerprint and key ID.

## Output Format

The generated key is intentionally minimal:

- one primary Ed25519 signing/certification key
- one User ID
- unencrypted secret-key material

The armored private key block is written to stdout, which makes it easy to redirect into a file or pipe into another command. With `--encrypt-to`, stdout carries an armored `PGP MESSAGE` encrypted to the recipient instead.

## Tuning

The first run on a given device performs a kernel auto-tuning pass (block size, global work size, and backend-specific compile options) and caches the result under `$XDG_CACHE_HOME/pgp-vanity/` (or the platform equivalent). Delete the `hip-tuning.txt` / `cuda-tuning.txt` file there to re-tune.

## Development

Run the test suite with:

```bash
cargo test
```

With Nix, prefix commands with the dev shell, and validate the packaged flake entrypoint with:

```bash
nix develop -c cargo test
nix flake check
```

The test suite includes:

- unit tests for prefix parsing and fingerprint helpers
- integration tests that compare the computed v4 fingerprint and key ID against GnuPG output
- integration tests that import the generated private key block into GnuPG
- a CLI integration test that exercises the GPU-backed search path

The GnuPG integration tests run when `gpg` is available in `PATH`, and the GPU-backed CLI test runs when a supported GPU is available.
