use std::io::Write;
use std::time::{Duration, UNIX_EPOCH};

use anyhow::{Result, ensure};
use sequoia_openpgp::Cert;
use sequoia_openpgp::packet::signature::SignatureBuilder;
use sequoia_openpgp::packet::{Key, Packet, UserID, key::Key4, key::PrimaryRole, key::SecretParts};
use sequoia_openpgp::parse::Parse;
use sequoia_openpgp::policy::StandardPolicy;
use sequoia_openpgp::serialize::Serialize;
use sequoia_openpgp::serialize::stream::{Armorer, Encryptor, LiteralWriter, Message};
use sequoia_openpgp::types::{KeyFlags, SignatureType};

pub fn build_armored_secret_key_block(
    seed: &[u8; 32],
    timestamp: u32,
    uid: &str,
) -> Result<String> {
    let creation_time = UNIX_EPOCH + Duration::from_secs(u64::from(timestamp));
    let key4: Key4<SecretParts, PrimaryRole> =
        Key4::import_secret_ed25519(seed, Some(creation_time))?;

    let secret_key: Key<SecretParts, PrimaryRole> = key4.into();
    let cert = Cert::try_from(vec![Packet::SecretKey(secret_key.clone())])?;
    let mut signer = secret_key.into_keypair()?;

    let user_id = UserID::from(uid);
    let binding_signature = user_id.bind(
        &mut signer,
        &cert,
        SignatureBuilder::new(SignatureType::PositiveCertification)
            .set_signature_creation_time(creation_time)?
            .set_key_flags(KeyFlags::empty().set_certification().set_signing())?,
    )?;

    let cert = cert
        .insert_packets(vec![Packet::from(user_id), Packet::from(binding_signature)])?
        .0;

    let mut armored = Vec::new();
    cert.as_tsk().armored().serialize(&mut armored)?;
    Ok(strip_armor_headers(String::from_utf8(armored)?))
}

pub fn encrypt_for_recipient(plaintext: &str, recipient_armor: &[u8]) -> Result<String> {
    let cert = Cert::from_bytes(recipient_armor)?;
    let policy = StandardPolicy::new();
    let recipients: Vec<_> = cert
        .keys()
        .with_policy(&policy, None)
        .supported()
        .alive()
        .revoked(false)
        .for_transport_encryption()
        .collect();

    ensure!(
        !recipients.is_empty(),
        "recipient certificate has no usable encryption-capable subkey"
    );

    let mut sink = Vec::new();
    {
        let message = Message::new(&mut sink);
        let message = Armorer::new(message).build()?;
        let message = Encryptor::for_recipients(message, recipients).build()?;
        let mut writer = LiteralWriter::new(message).build()?;
        writer.write_all(plaintext.as_bytes())?;
        writer.finalize()?;
    }

    Ok(strip_armor_headers(String::from_utf8(sink)?))
}

fn strip_armor_headers(armored: String) -> String {
    let mut lines = armored.lines();
    let mut stripped = String::new();

    if let Some(begin) = lines.next() {
        stripped.push_str(begin);
        stripped.push('\n');
    }

    let mut in_headers = true;
    for line in lines {
        if in_headers {
            if line.is_empty() {
                stripped.push('\n');
                in_headers = false;
            }
            continue;
        }

        stripped.push_str(line);
        stripped.push('\n');
    }

    stripped
}
