pub mod crypto;
pub mod models;
pub mod scoring;

pub use crypto::{
    compute_attestation_id, compute_evidence_root, compute_hash, issue_attestation,
    verify_attestation, AttestationError, CryptoError,
};
pub use models::{Attestation, AttestationPayload, DimensionScores, EvidenceBundle, IhsanScore};
pub use scoring::{calculate_poi, verify_ihsan, ScoringError};

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn test_poi_calculation() {
        let bundle = EvidenceBundle {
            content_hash: "dummy".to_string(),
            metadata: Default::default(),
            dimensions: DimensionScores {
                quality: 1.0,
                utility: 1.0,
                trust: 1.0,
                fairness: 1.0,
                diversity: 1.0,
            },
        };
        // Weights sum to 1.0. Raw POI = 1.0. Penalty = 0. Final = 1.0.
        let poi = calculate_poi(&bundle).unwrap();
        assert!((poi - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ihsan_fail_closed() {
        let score = IhsanScore {
            truthfulness: 0.90,
            dignity: 0.90,
            fairness: 0.90,
            excellence: 0.90,
            sustainability: 0.90,
        };
        // Total = 0.90 < 0.95. Should fail.
        let result = verify_ihsan(&score);
        assert!(result.is_err());
    }

    #[test]
    fn test_ihsan_pass() {
        let score = IhsanScore {
            truthfulness: 1.0,
            dignity: 1.0,
            fairness: 1.0,
            excellence: 1.0,
            sustainability: 1.0,
        };
        assert!(verify_ihsan(&score).is_ok());
    }

    #[test]
    fn test_issue_and_verify_attestation() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let bundle = EvidenceBundle {
            content_hash: "dummy".to_string(),
            metadata: Default::default(),
            dimensions: DimensionScores {
                quality: 1.0,
                utility: 1.0,
                trust: 1.0,
                fairness: 1.0,
                diversity: 1.0,
            },
        };

        let attestation = issue_attestation("contributor", 42, &bundle, &signing_key, 0.99).unwrap();
        assert!(verify_attestation(&attestation, &bundle).is_ok());
    }

    #[test]
    fn test_verify_attestation_detects_mismatch() {
        let signing_key = SigningKey::from_bytes(&[9u8; 32]);
        let bundle = EvidenceBundle {
            content_hash: "dummy".to_string(),
            metadata: Default::default(),
            dimensions: DimensionScores {
                quality: 0.9,
                utility: 0.9,
                trust: 0.9,
                fairness: 0.9,
                diversity: 0.9,
            },
        };

        let mut attestation = issue_attestation("contributor", 7, &bundle, &signing_key, 0.99).unwrap();
        attestation.poi_score = 0.1;
        assert!(verify_attestation(&attestation, &bundle).is_err());
    }

    #[test]
    fn test_validation_threshold_enforced() {
        let signing_key = SigningKey::from_bytes(&[11u8; 32]);
        let bundle = EvidenceBundle {
            content_hash: "dummy".to_string(),
            metadata: Default::default(),
            dimensions: DimensionScores {
                quality: 0.9,
                utility: 0.9,
                trust: 0.9,
                fairness: 0.9,
                diversity: 0.9,
            },
        };

        let err = issue_attestation("contributor", 9, &bundle, &signing_key, 0.5).unwrap_err();
        assert!(matches!(err, AttestationError::ValidationBelowThreshold));
    }

    #[test]
    fn test_poi_rejects_non_finite() {
        let bundle = EvidenceBundle {
            content_hash: "dummy".to_string(),
            metadata: Default::default(),
            dimensions: DimensionScores {
                quality: f64::NAN,
                utility: 1.0,
                trust: 1.0,
                fairness: 1.0,
                diversity: 1.0,
            },
        };

        assert!(calculate_poi(&bundle).is_err());
    }

    #[test]
    fn test_ihsan_rejects_non_finite() {
        let score = IhsanScore {
            truthfulness: 1.0,
            dignity: f64::INFINITY,
            fairness: 1.0,
            excellence: 1.0,
            sustainability: 1.0,
        };

        assert!(verify_ihsan(&score).is_err());
    }
}
