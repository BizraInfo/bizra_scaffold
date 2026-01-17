//! BIZRA JCS - RFC 8785 JSON Canonicalization Scheme
//!
//! This crate implements JSON Canonicalization Scheme (JCS) as defined in RFC 8785
//! for deterministic JSON serialization. This is critical for:
//!
//! - Receipt hashing (same input → same hash across all platforms)
//! - Signature verification (canonical form for signing)
//! - Audit trail integrity (reproducible evidence)
//!
//! # RFC 8785 Summary
//!
//! 1. **Object keys**: Sorted by Unicode code points (lexicographic)
//! 2. **No whitespace**: No spaces between elements
//! 3. **Numbers**: No leading zeros, no unnecessary decimal points
//! 4. **Strings**: Minimal UTF-8 escaping
//!
//! # Example
//!
//! ```rust
//! use bizra_jcs::{canonicalize, hash_canonical};
//! use serde_json::json;
//!
//! let value = json!({"z": 1, "a": 2});
//! let canonical = canonicalize(&value).unwrap();
//! assert_eq!(canonical, r#"{"a":2,"z":1}"#);
//!
//! let hash = hash_canonical(&value).unwrap();
//! // Hash is deterministic across all platforms
//! ```

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Errors that can occur during JCS canonicalization
#[derive(Debug, Error)]
pub enum JcsError {
    #[error("JSON serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid number format: {0}")]
    InvalidNumber(String),
}

/// Result type for JCS operations
pub type JcsResult<T> = Result<T, JcsError>;

/// Canonicalize a serde_json::Value according to RFC 8785
///
/// # Rules Applied
///
/// 1. Object keys sorted lexicographically by Unicode code points
/// 2. No whitespace between tokens
/// 3. Numbers serialized without unnecessary formatting
/// 4. Strings with minimal escaping
pub fn canonicalize_value(value: &serde_json::Value) -> JcsResult<String> {
    match value {
        serde_json::Value::Null => Ok("null".to_string()),

        serde_json::Value::Bool(b) => Ok(if *b { "true" } else { "false" }.to_string()),

        serde_json::Value::Number(n) => {
            // RFC 8785: Numbers must not have unnecessary formatting
            // - No leading zeros (except for "0" itself)
            // - No trailing zeros after decimal
            // - No positive sign
            // - Exponential notation only when necessary

            if let Some(i) = n.as_i64() {
                Ok(i.to_string())
            } else if let Some(u) = n.as_u64() {
                Ok(u.to_string())
            } else if let Some(f) = n.as_f64() {
                // WARNING: Floats in receipts violate determinism!
                // This path exists for completeness but should trigger warnings.
                // Use Fixed64 (i64) instead.
                canonicalize_float(f)
            } else {
                Err(JcsError::InvalidNumber(n.to_string()))
            }
        }

        serde_json::Value::String(s) => {
            // RFC 8785: Strings use minimal escaping
            // Only escape: " \ and control characters (0x00-0x1F)
            Ok(escape_string(s))
        }

        serde_json::Value::Array(arr) => {
            let items: JcsResult<Vec<String>> = arr.iter().map(canonicalize_value).collect();
            Ok(format!("[{}]", items?.join(",")))
        }

        serde_json::Value::Object(obj) => {
            // RFC 8785: Keys sorted by Unicode code points
            let mut keys: Vec<&String> = obj.keys().collect();
            keys.sort_by(|a, b| {
                // Sort by Unicode code points (byte comparison for ASCII)
                a.as_bytes().cmp(b.as_bytes())
            });

            let pairs: JcsResult<Vec<String>> = keys
                .iter()
                .map(|k| {
                    let v = canonicalize_value(&obj[*k])?;
                    Ok(format!("{}:{}", escape_string(k), v))
                })
                .collect();

            Ok(format!("{{{}}}", pairs?.join(",")))
        }
    }
}

/// Canonicalize a float according to RFC 8785
///
/// Note: Using floats in consensus-critical code is discouraged.
/// Prefer Fixed64 (i64 with implicit scale) for determinism.
fn canonicalize_float(f: f64) -> JcsResult<String> {
    if !f.is_finite() {
        return Err(JcsError::InvalidNumber(format!(
            "Non-finite number: {}",
            f
        )));
    }

    // Special case: -0 becomes 0
    if f == 0.0 {
        return Ok("0".to_string());
    }

    // Use JavaScript-compatible representation
    // This matches ECMAScript NumberToString
    let s = format!("{}", f);

    // Remove unnecessary trailing zeros and decimal point
    let s = if s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        &s
    };

    Ok(s.to_string())
}

/// Escape a string according to RFC 8785 / JSON rules
fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    result.push('"');

    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\x08' => result.push_str("\\b"),
            '\x0c' => result.push_str("\\f"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                // Escape other control characters as \uXXXX
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }

    result.push('"');
    result
}

/// Canonicalize any serializable value
///
/// Converts to serde_json::Value first, then applies JCS rules.
pub fn canonicalize<T: Serialize>(value: &T) -> JcsResult<String> {
    let json_value = serde_json::to_value(value)?;
    canonicalize_value(&json_value)
}

/// Compute SHA-256 hash of JCS-canonicalized value
///
/// Returns lowercase hex string.
pub fn hash_canonical<T: Serialize>(value: &T) -> JcsResult<String> {
    let canonical = canonicalize(value)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute SHA-256 hash of JCS-canonicalized serde_json::Value
pub fn hash_value(value: &serde_json::Value) -> JcsResult<String> {
    let canonical = canonicalize_value(value)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify that a hash matches the canonical form of a value
pub fn verify_hash<T: Serialize>(value: &T, expected_hash: &str) -> JcsResult<bool> {
    let computed = hash_canonical(value)?;
    Ok(computed == expected_hash)
}

/// Compute SHA-256 digest of JCS-canonicalized value
///
/// Returns raw 32-byte digest (for signing).
pub fn compute_digest<T: Serialize>(value: &T) -> JcsResult<[u8; 32]> {
    let canonical = canonicalize(value)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    Ok(hasher.finalize().into())
}

/// Compute payload ID: base64url(sha256(JCS(payload)))
///
/// This creates a unique, deterministic identifier for any payload.
/// Used for receipt IDs and cross-references.
pub fn compute_payload_id<T: Serialize>(value: &T) -> JcsResult<String> {
    let digest = compute_digest(value)?;
    Ok(base64url_encode(&digest))
}

/// Base64url encode without padding (RFC 4648 §5)
fn base64url_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    let mut result = String::with_capacity((data.len() * 4 + 2) / 3);

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        }
        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        }
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_key_ordering() {
        let value = json!({"z": 1, "a": 2, "m": 3});
        let canonical = canonicalize_value(&value).unwrap();
        assert_eq!(canonical, r#"{"a":2,"m":3,"z":1}"#);
    }

    #[test]
    fn test_no_whitespace() {
        let value = json!({"key": [1, 2, 3]});
        let canonical = canonicalize_value(&value).unwrap();
        assert_eq!(canonical, r#"{"key":[1,2,3]}"#);
    }

    #[test]
    fn test_nested_objects() {
        let value = json!({"b": {"z": 1, "a": 2}, "a": 1});
        let canonical = canonicalize_value(&value).unwrap();
        assert_eq!(canonical, r#"{"a":1,"b":{"a":2,"z":1}}"#);
    }

    #[test]
    fn test_string_escaping() {
        let value = json!({"msg": "hello\nworld"});
        let canonical = canonicalize_value(&value).unwrap();
        assert_eq!(canonical, r#"{"msg":"hello\nworld"}"#);
    }

    #[test]
    fn test_integers() {
        let value = json!({"num": 42});
        let canonical = canonicalize_value(&value).unwrap();
        assert_eq!(canonical, r#"{"num":42}"#);
    }

    #[test]
    fn test_negative_integers() {
        let value = json!({"num": -123});
        let canonical = canonicalize_value(&value).unwrap();
        assert_eq!(canonical, r#"{"num":-123}"#);
    }

    #[test]
    fn test_boolean() {
        let value = json!({"flag": true, "other": false});
        let canonical = canonicalize_value(&value).unwrap();
        assert_eq!(canonical, r#"{"flag":true,"other":false}"#);
    }

    #[test]
    fn test_null() {
        let value = json!({"val": null});
        let canonical = canonicalize_value(&value).unwrap();
        assert_eq!(canonical, r#"{"val":null}"#);
    }

    #[test]
    fn test_determinism() {
        // Same input must produce same output every time
        let value = json!({
            "session": "sess-001",
            "counter": 42,
            "nonce": "abc123"
        });

        let hash1 = hash_canonical(&value).unwrap();
        let hash2 = hash_canonical(&value).unwrap();
        let hash3 = hash_canonical(&value).unwrap();

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_verify_hash() {
        let value = json!({"test": 123});
        let hash = hash_canonical(&value).unwrap();
        assert!(verify_hash(&value, &hash).unwrap());
        assert!(!verify_hash(&value, "wrong_hash").unwrap());
    }

    #[test]
    fn test_unicode_key_ordering() {
        // RFC 8785: Sort by Unicode code points
        let value = json!({"β": 2, "α": 1, "a": 0});
        let canonical = canonicalize_value(&value).unwrap();
        // ASCII 'a' (0x61) comes before Greek letters (0x03B1, 0x03B2)
        assert!(canonical.starts_with(r#"{"a":0"#));
    }

    #[test]
    fn test_empty_structures() {
        assert_eq!(canonicalize_value(&json!({})).unwrap(), "{}");
        assert_eq!(canonicalize_value(&json!([])).unwrap(), "[]");
    }

    #[test]
    fn test_large_integers() {
        // Ensure i64 range is handled
        let value = json!({"big": 9007199254740991_i64}); // 2^53 - 1
        let canonical = canonicalize_value(&value).unwrap();
        assert_eq!(canonical, r#"{"big":9007199254740991}"#);
    }
}
