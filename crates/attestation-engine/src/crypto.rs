use blake3::Hasher;
use ed25519_dalek::{Signature, SignatureError, SigningKey, VerifyingKey};
use ed25519_dalek::{Signer, Verifier};
use hex::FromHex;
use serde::Serialize;
use thiserror::Error;

use crate::models::{Attestation, AttestationPayload, EvidenceBundle};
use crate::scoring::{calculate_poi, ScoringError};

const POI_TOLERANCE: f64 = 1e-6;
const VALIDATION_THRESHOLD: f64 = 0.85;

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Invalid hex: {0}")]
    InvalidHex(#[from] hex::FromHexError),
    #[error("Invalid signature: {0}")]
    InvalidSignature(#[from] SignatureError),
    #[error("Invalid public key")]
    InvalidPublicKey,
}

#[derive(Error, Debug)]
pub enum AttestationError {
    #[error("Crypto error: {0}")]
    Crypto(#[from] CryptoError),
    #[error("Scoring error: {0}")]
    Scoring(#[from] ScoringError),
    #[error("Evidence root mismatch")]
    EvidenceRootMismatch,
    #[error("Attestation id mismatch")]
    AttestationIdMismatch,
    #[error("PoI mismatch (expected {expected}, got {actual})")]
    PoiMismatch { expected: f64, actual: f64 },
    #[error("PoI score is not finite")]
    InvalidPoiScore,
    #[error("Validation score out of range [0,1]")]
    InvalidValidationScore,
    #[error("Validation score below required threshold")]
    ValidationBelowThreshold,
    #[error("Signature verification failed")]
    SignatureInvalid,
}

fn canonical_json<T: Serialize>(data: &T) -> Result<Vec<u8>, CryptoError> {
    serde_jcs::to_vec(data).map_err(|e| CryptoError::Serialization(e))
}

/// Computes the Blake3 hash of a data structure using canonical JSON serialization.
pub fn compute_hash<T: Serialize>(data: &T) -> Result<String, CryptoError> {
    let json_bytes = canonical_json(data)?;
    let mut hasher = Hasher::new();
    hasher.update(&json_bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

pub fn compute_evidence_root(evidence: &EvidenceBundle) -> Result<String, CryptoError> {
    compute_hash(evidence)
}

pub fn compute_attestation_id(contributor: &str, epoch: u64, evidence_root: &str) -> String {
    let mut hasher = Hasher::new();
    hasher.update(contributor.as_bytes());
    hasher.update(&epoch.to_be_bytes());
    hasher.update(evidence_root.as_bytes());
    hasher.finalize().to_hex().to_string()
}

pub fn verifying_key_from_hex(hex_str: &str) -> Result<VerifyingKey, CryptoError> {
    let bytes = Vec::from_hex(hex_str)?;
    let key_bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| CryptoError::InvalidPublicKey)?;
    VerifyingKey::from_bytes(&key_bytes).map_err(|_| CryptoError::InvalidPublicKey)
}

pub fn public_key_hex(signing_key: &SigningKey) -> String {
    hex::encode(signing_key.verifying_key().to_bytes())
}

pub fn sign_payload(
    payload: &AttestationPayload,
    signing_key: &SigningKey,
) -> Result<String, CryptoError> {
    let payload_bytes = canonical_json(payload)?;
    let signature = signing_key.sign(&payload_bytes);
    Ok(hex::encode(signature.to_bytes()))
}

pub fn verify_payload_signature(
    payload: &AttestationPayload,
    signature_hex: &str,
    verifying_key: &VerifyingKey,
) -> Result<(), CryptoError> {
    let payload_bytes = canonical_json(payload)?;
    let signature_bytes = Vec::from_hex(signature_hex)?;
    let signature = Signature::from_slice(&signature_bytes)?;
    verifying_key.verify(&payload_bytes, &signature)?;
    Ok(())
}

pub fn issue_attestation(
    contributor: &str,
    epoch: u64,
    evidence: &EvidenceBundle,
    validator_signing_key: &SigningKey,
    validation_score: f64,
) -> Result<Attestation, AttestationError> {
    validate_validation_score(validation_score)?;

    let evidence_root = compute_evidence_root(evidence)?;
    let poi_score = calculate_poi(evidence)?;
    let attestation_id = compute_attestation_id(contributor, epoch, &evidence_root);
    let validator = public_key_hex(validator_signing_key);

    let payload = AttestationPayload {
        attestation_id,
        contributor: contributor.to_string(),
        validator,
        epoch,
        evidence_root,
        poi_score,
        validation_score,
    };

    let signature = sign_payload(&payload, validator_signing_key)?;
    Ok(payload.into_attestation(signature))
}

pub fn verify_attestation(
    attestation: &Attestation,
    evidence: &EvidenceBundle,
) -> Result<(), AttestationError> {
    validate_validation_score(attestation.validation_score)?;

    let evidence_root = compute_evidence_root(evidence)?;
    if evidence_root != attestation.evidence_root {
        return Err(AttestationError::EvidenceRootMismatch);
    }

    let expected_id = compute_attestation_id(
        &attestation.contributor,
        attestation.epoch,
        &attestation.evidence_root,
    );
    if expected_id != attestation.attestation_id {
        return Err(AttestationError::AttestationIdMismatch);
    }

    if !attestation.poi_score.is_finite() {
        return Err(AttestationError::InvalidPoiScore);
    }

    let expected_poi = calculate_poi(evidence)?;
    if (expected_poi - attestation.poi_score).abs() > POI_TOLERANCE {
        return Err(AttestationError::PoiMismatch {
            expected: expected_poi,
            actual: attestation.poi_score,
        });
    }

    let verifying_key = verifying_key_from_hex(&attestation.validator)?;
    let payload = attestation.payload();
    match verify_payload_signature(&payload, &attestation.signature, &verifying_key) {
        Ok(()) => {}
        Err(CryptoError::InvalidSignature(_)) => return Err(AttestationError::SignatureInvalid),
        Err(err) => return Err(AttestationError::Crypto(err)),
    }

    Ok(())
}

fn validate_validation_score(value: f64) -> Result<(), AttestationError> {
    if !value.is_finite() || value < 0.0 || value > 1.0 {
        return Err(AttestationError::InvalidValidationScore);
    }
    if value < VALIDATION_THRESHOLD {
        return Err(AttestationError::ValidationBelowThreshold);
    }
    Ok(())
}
