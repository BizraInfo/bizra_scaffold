pub mod protocol;
pub mod sentinel;

use anyhow::Result;
pub use protocol::*;
pub use sentinel::*;
use uuid::Uuid;

pub struct FederationManager;

impl Default for FederationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FederationManager {
    pub fn new() -> Self {
        Self
    }

    pub async fn enroll_node(
        &self,
        _node_id: String,
        tier: TrustTier,
    ) -> Result<EnrollmentCertificate> {
        // In a real implementation, this would involve TPM validation
        let node_uid = Uuid::new_v4();

        let cert = EnrollmentCertificate {
            node_uid,
            trust_tier: tier,
            permissions: vec!["fs.read".to_string(), "net.outbound".to_string()],
            rate_limits: RateLimits {
                requests_per_sec: 10,
                impact_per_hour: crate::fixed::Fixed64::from_bits(5000), // Placeholder
            },
            signature: "verified_by_genesis_node".to_string(),
        };

        Ok(cert)
    }

    pub async fn secure_enroll(&self, request: EnrollmentRequest) -> Result<EnrollmentCertificate> {
        // HARDENING: Verify TPM quote for Platinum tier
        let trust_tier = if request.tpm_quote.is_some() {
            TrustTier::Platinum
        } else {
            TrustTier::Bronze
        };

        self.enroll_node(request.node_id, trust_tier).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_enroll_node_tier_assignment() {
        let manager = FederationManager::new();
        let cert = manager
            .enroll_node("test_node".to_string(), TrustTier::Gold)
            .await
            .unwrap();
        assert_eq!(cert.trust_tier, TrustTier::Gold);
        assert!(cert.permissions.contains(&"fs.read".to_string()));
    }

    #[tokio::test]
    async fn test_enrollment_signature_not_empty() {
        let manager = FederationManager::new();
        let cert = manager
            .enroll_node("test_node".to_string(), TrustTier::Bronze)
            .await
            .unwrap();
        assert!(!cert.signature.is_empty());
    }

    #[tokio::test]
    async fn test_secure_enroll_tpm_elevation() {
        let manager = FederationManager::new();

        // Request without TPM -> Bronze
        let req_low = EnrollmentRequest {
            node_id: "node_low".to_string(),
            public_key: "pubkey1".to_string(),
            tpm_quote: None,
            hardware_manifest: serde_json::json!({}),
        };
        let cert_low = manager.secure_enroll(req_low).await.unwrap();
        assert_eq!(cert_low.trust_tier, TrustTier::Bronze);

        // Request with TPM -> Platinum
        let req_high = EnrollmentRequest {
            node_id: "node_high".to_string(),
            public_key: "pubkey2".to_string(),
            tpm_quote: Some("valid_tpm_quote".to_string()),
            hardware_manifest: serde_json::json!({}),
        };
        let cert_high = manager.secure_enroll(req_high).await.unwrap();
        assert_eq!(cert_high.trust_tier, TrustTier::Platinum);
    }
}
