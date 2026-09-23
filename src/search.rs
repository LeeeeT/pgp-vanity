use std::io::{self, Write};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, bail, ensure};
use ed25519_dalek::SigningKey;
use rand::TryRngCore;
use rand::rngs::OsRng;

use crate::fingerprint::{FingerprintSearch, key_id_from_fingerprint};
use crate::gpu::GpuSearchEngine;
use crate::{HexPrefixSet, hex_upper, key_id_hex};

const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub prefixes: HexPrefixSet,
    pub fixed_seed: Option<[u8; 32]>,
    pub max_key_attempts: Option<u64>,
    pub progress: bool,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub seed: [u8; 32],
    pub public_key: [u8; 32],
    pub timestamp: u32,
    pub fingerprint: [u8; 20],
    pub key_id: u64,
    pub key_attempts: u64,
    pub timestamps_checked: u64,
    pub elapsed: Duration,
}

impl SearchResult {
    pub fn key_id_hex(&self) -> String {
        key_id_hex(self.key_id)
    }

    pub fn fingerprint_hex(&self) -> String {
        hex_upper(self.fingerprint)
    }

    pub fn seed_hex(&self) -> String {
        hex_upper(self.seed)
    }

    pub fn public_key_hex(&self) -> String {
        hex_upper(self.public_key)
    }
}

pub fn search(config: SearchConfig) -> Result<SearchResult> {
    let mut engine = GpuSearchEngine::new()?;
    ensure!(
        engine.batch_size() > 0,
        "GPU batch size must be greater than zero"
    );

    eprintln!(
        "using {} GPU: {}",
        engine.backend_name(),
        engine.device_name()
    );

    engine.prepare_prefixes(&config.prefixes)?;
    let prefix_display = format_prefix_list(&config.prefixes);
    let timestamp_count = timestamp_count_through(SystemTime::now())?;

    let overall_start = Instant::now();
    let mut total_checked = 0u64;
    let mut key_attempts = 0u64;
    let mut progress_line_len = 0usize;

    loop {
        key_attempts += 1;

        if let Some(limit) = config.max_key_attempts
            && key_attempts > limit
        {
            if config.progress {
                clear_progress_line(progress_line_len);
            }
            bail!("no key ID starting with {prefix_display} found after {limit} key attempt(s)");
        }

        let seed = match config.fixed_seed {
            Some(seed) => seed,
            None => random_seed()?,
        };

        let signing_key = SigningKey::from_bytes(&seed);
        let public_key = signing_key.verifying_key().to_bytes();
        let searcher = FingerprintSearch::new(public_key);
        engine.prepare_search(&searcher.base_words())?;

        let mut last_report_total = total_checked;
        let mut last_report = Instant::now();
        let expected_candidates = config.prefixes.search_space_size();
        let mut batch_start = 0u64;
        let mut progress_shown_this_attempt = false;

        while batch_start < timestamp_count {
            let batch_count = engine.batch_size().min(timestamp_count - batch_start) as u32;
            let found_timestamp = engine.search_batch(batch_start as u32, batch_count)?;

            let checked_in_batch = match found_timestamp {
                Some(timestamp) => {
                    ensure!(
                        u64::from(timestamp) >= batch_start
                            && u64::from(timestamp) < batch_start + u64::from(batch_count),
                        "GPU returned an out-of-range timestamp"
                    );
                    u64::from(timestamp) - batch_start + 1
                }
                None => u64::from(batch_count),
            };

            total_checked += checked_in_batch;

            if config.progress
                && (!progress_shown_this_attempt || last_report.elapsed() >= PROGRESS_INTERVAL)
            {
                let now = Instant::now();
                let elapsed_since_last_report = now.saturating_duration_since(last_report);
                let processed_since_last_report = total_checked.saturating_sub(last_report_total);
                let rate = if elapsed_since_last_report.is_zero() {
                    0.0
                } else {
                    processed_since_last_report as f64 / elapsed_since_last_report.as_secs_f64()
                };

                let remaining = expected_candidates.saturating_sub(u128::from(total_checked));
                let eta = if rate > 0.0 {
                    format_duration_human(remaining as f64 / rate)
                } else {
                    "unknown".to_string()
                };

                let line = format!(
                    "searching {prefix_display} on {}: {}% of {} combinations explored | {:.2} GH/s | ETA {eta}",
                    engine.device_name(),
                    format_percentage(total_checked as f64 / expected_candidates as f64 * 100.0),
                    format_with_commas(expected_candidates),
                    rate / 1_000_000_000.0
                );
                render_progress_line(&line, &mut progress_line_len);
                last_report = now;
                last_report_total = total_checked;
                progress_shown_this_attempt = true;
            }

            if let Some(timestamp) = found_timestamp {
                if config.progress {
                    clear_progress_line(progress_line_len);
                }

                let fingerprint = searcher.fingerprint(timestamp);
                let key_id = key_id_from_fingerprint(&fingerprint);
                ensure!(
                    config.prefixes.matches(key_id),
                    "GPU result did not match any requested key ID prefix"
                );

                return Ok(SearchResult {
                    seed,
                    public_key,
                    timestamp,
                    fingerprint,
                    key_id,
                    key_attempts,
                    timestamps_checked: total_checked,
                    elapsed: overall_start.elapsed(),
                });
            }

            batch_start += u64::from(batch_count);
        }

        if config.fixed_seed.is_some() {
            if config.progress {
                clear_progress_line(progress_line_len);
            }
            bail!(
                "searched all {timestamp_count} non-future timestamps for the provided seed without finding prefix {prefix_display}"
            );
        }
    }
}

fn format_prefix_list(prefixes: &HexPrefixSet) -> String {
    prefixes.normalized_list().join(", ")
}

fn random_seed() -> Result<[u8; 32]> {
    let mut seed = [0u8; 32];
    OsRng
        .try_fill_bytes(&mut seed)
        .map_err(|error| anyhow::anyhow!("failed to read secure randomness: {error}"))?;
    Ok(seed)
}

fn timestamp_count_through(now: SystemTime) -> Result<u64> {
    let current_timestamp = now
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs()
        .min(u64::from(u32::MAX));
    Ok(current_timestamp + 1)
}

fn render_progress_line(line: &str, progress_line_len: &mut usize) {
    let padding = progress_line_len.saturating_sub(line.len());
    eprint!("\r{line}{}", " ".repeat(padding));
    let _ = io::stderr().flush();
    *progress_line_len = line.len();
}

fn clear_progress_line(previous_len: usize) {
    if previous_len == 0 {
        return;
    }

    eprint!("\r{}\r", " ".repeat(previous_len));
    let _ = io::stderr().flush();
}

fn format_with_commas(value: u128) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);

    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(digit);
    }

    formatted
}

fn format_duration_human(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "unknown".to_string();
    }

    let total = seconds.round() as u64;
    if total == 0 {
        return "0s".to_string();
    }

    const YEAR: u64 = 365 * DAY;
    const DAY: u64 = 24 * HOUR;
    const HOUR: u64 = 60 * MINUTE;
    const MINUTE: u64 = 60;

    let units = [
        ("y", YEAR),
        ("d", DAY),
        ("h", HOUR),
        ("m", MINUTE),
        ("s", 1),
    ];

    let mut remaining = total;
    let mut parts = Vec::new();
    for (suffix, size) in units {
        let count = remaining / size;
        if count > 0 {
            parts.push(format!("{count}{suffix}"));
            remaining %= size;
        }
        // Stop once we have the two most significant non-zero units.
        if parts.len() == 2 {
            break;
        }
    }

    parts.join(" ")
}

fn format_percentage(value: f64) -> String {
    if value >= 10.0 {
        format!("{value:.2}")
    } else if value >= 0.01 {
        format!("{value:.4}")
    } else {
        format!("{value:.6}")
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::timestamp_count_through;

    #[test]
    fn timestamp_count_includes_now_but_not_the_future() {
        assert_eq!(timestamp_count_through(UNIX_EPOCH).unwrap(), 1);
        assert_eq!(
            timestamp_count_through(UNIX_EPOCH + Duration::from_secs(42)).unwrap(),
            43
        );
        assert_eq!(
            timestamp_count_through(UNIX_EPOCH + Duration::from_secs(u64::from(u32::MAX) + 1))
                .unwrap(),
            u64::from(u32::MAX) + 1
        );
    }
}
