// src/fixed.rs - Fixed-Point Arithmetic for Deterministic Computation
// Q32.32 format: 32 bits integer, 32 bits fraction
// Critical for institutional-grade bare-metal/UEFI deployments where f64 is non-deterministic.

use core::fmt;
use core::iter::Sum;
use core::ops::{Add, Div, Mul, Sub};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Q32.32 Fixed-Point Number
/// Represents values in range [-2^31, 2^31) with precision of 2^-32
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Fixed64(i64);

impl Fixed64 {
    /// Fractional bits (32 bits of precision)
    pub const FRAC_BITS: u32 = 32;
    pub const SCALE: i64 = 1i64 << Self::FRAC_BITS;

    /// Constants
    pub const ZERO: Fixed64 = Fixed64(0);
    pub const ONE: Fixed64 = Fixed64(Self::SCALE);
    pub const HALF: Fixed64 = Fixed64(Self::SCALE / 2);

    /// Create from integer
    #[inline]
    pub const fn from_int(n: i32) -> Self {
        Fixed64((n as i64) << Self::FRAC_BITS)
    }

    /// Create from i64 (for large whole numbers)
    #[inline]
    pub const fn from_i64(n: i64) -> Self {
        Fixed64(n << Self::FRAC_BITS)
    }

    /// Create from raw bits (internal representation)
    #[inline]
    pub const fn from_bits(bits: i64) -> Self {
        Fixed64(bits)
    }

    /// Get raw bits
    #[inline]
    pub const fn to_bits(self) -> i64 {
        self.0
    }

    /// Create from f64 (for initialization only - not for hot path)
    #[inline]
    pub fn from_f64(f: f64) -> Self {
        Fixed64((f * Self::SCALE as f64) as i64)
    }

    /// Convert to f64 (for display/logging only - not for computation)
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    /// Integer part
    #[inline]
    pub const fn int_part(self) -> i32 {
        (self.0 >> Self::FRAC_BITS) as i32
    }

    /// Fractional part (as raw bits)
    #[inline]
    pub const fn frac_part(self) -> u32 {
        (self.0 & (Self::SCALE - 1)) as u32
    }

    /// Saturating addition (no overflow panic)
    #[inline]
    pub fn saturating_add(self, other: Self) -> Self {
        Fixed64(self.0.saturating_add(other.0))
    }

    /// Saturating subtraction
    #[inline]
    pub fn saturating_sub(self, other: Self) -> Self {
        Fixed64(self.0.saturating_sub(other.0))
    }

    /// Saturating multiplication (handles overflow)
    #[inline]
    pub fn saturating_mul(self, other: Self) -> Self {
        // Use i128 for intermediate to prevent overflow
        let result = (self.0 as i128 * other.0 as i128) >> Self::FRAC_BITS;
        if result > i64::MAX as i128 {
            Fixed64(i64::MAX)
        } else if result < i64::MIN as i128 {
            Fixed64(i64::MIN)
        } else {
            Fixed64(result as i64)
        }
    }

    /// Division (panics on divide by zero)
    #[inline]
    pub fn checked_div(self, other: Self) -> Option<Self> {
        if other.0 == 0 {
            None
        } else {
            let result = ((self.0 as i128) << Self::FRAC_BITS) / other.0 as i128;
            if result > i64::MAX as i128 {
                Some(Fixed64(i64::MAX))
            } else if result < i64::MIN as i128 {
                Some(Fixed64(i64::MIN))
            } else {
                Some(Fixed64(result as i64))
            }
        }
    }

    /// Saturating division
    #[inline]
    pub fn saturating_div(self, other: Self) -> Self {
        self.checked_div(other).unwrap_or(if self.0 >= 0 {
            Fixed64(i64::MAX)
        } else {
            Fixed64(i64::MIN)
        })
    }

    /// Minimum of two values
    #[inline]
    pub fn min(self, other: Self) -> Self {
        if self.0 <= other.0 {
            self
        } else {
            other
        }
    }

    /// Maximum of two values
    #[inline]
    pub fn max(self, other: Self) -> Self {
        if self.0 >= other.0 {
            self
        } else {
            other
        }
    }

    /// Absolute value
    #[inline]
    pub fn abs(self) -> Self {
        if self.0 < 0 {
            Fixed64(self.0.wrapping_neg())
        } else {
            self
        }
    }

    /// Clamp to range [min, max]
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    /// Integer power
    pub fn powi(mut self, mut exp: i32) -> Self {
        if exp == 0 {
            return Self::ONE;
        }
        if exp < 0 {
            return Self::ONE / self.powi(-exp);
        }
        let mut res = Self::ONE;
        while exp > 1 {
            if exp % 2 == 1 {
                res = res * self;
            }
            self = self * self;
            exp /= 2;
        }
        res * self
    }
}

// Operator implementations
impl Add for Fixed64 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Fixed64(self.0.wrapping_add(other.0))
    }
}

impl Sub for Fixed64 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Fixed64(self.0.wrapping_sub(other.0))
    }
}

impl Mul for Fixed64 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        self.saturating_mul(other)
    }
}

impl Div for Fixed64 {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        self.checked_div(other).expect("Division by zero")
    }
}

impl Sum for Fixed64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Fixed64::ZERO, |a, b| a + b)
    }
}

impl fmt::Display for Fixed64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.to_f64())
    }
}

impl Serialize for Fixed64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(self.to_f64())
    }
}

impl<'de> Deserialize<'de> for Fixed64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Fixed64Visitor;

        impl<'de> Visitor<'de> for Fixed64Visitor {
            type Value = Fixed64;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a fixed-point number")
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Fixed64::from_f64(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(Fixed64::from_i64(value))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value > i64::MAX as u64 {
                    return Err(E::custom("Fixed64 out of range"));
                }
                Ok(Fixed64::from_i64(value as i64))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let parsed: f64 = value.parse().map_err(E::custom)?;
                Ok(Fixed64::from_f64(parsed))
            }
        }

        deserializer.deserialize_any(Fixed64Visitor)
    }
}

pub mod serde_bits {
    use super::Fixed64;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &Fixed64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_i64(value.to_bits())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Fixed64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bits = i64::deserialize(deserializer)?;
        Ok(Fixed64::from_bits(bits))
    }
}

/// Institutional thresholds as fixed-point constants
pub mod thresholds {
    use super::Fixed64;

    /// Ihsan minimum threshold: 0.99 (Q32.32: 0x0_FD70A3D7)
    pub const IHSAN_MIN: Fixed64 = Fixed64::from_bits(0x0_FD70A3D7);

    /// SNR minimum threshold: 1.5 (Q32.32: 0x1_80000000)
    pub const SNR_MIN: Fixed64 = Fixed64::from_bits(0x1_80000000);

    /// Consensus quorum: 0.70 (Q32.32: 0x0_B3333333)
    pub const CONSENSUS_QUORUM: Fixed64 = Fixed64::from_bits(0x0_B3333333);

    /// Thermal limit: 85.0°C (Q32.32: 0x55_00000000)
    pub const THERMAL_LIMIT: Fixed64 = Fixed64::from_bits(0x55_00000000);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a = Fixed64::from_int(10);
        let b = Fixed64::from_int(3);

        assert_eq!((a + b).int_part(), 13);
        assert_eq!((a - b).int_part(), 7);
        assert_eq!((a * b).int_part(), 30);
        assert_eq!((a / b).int_part(), 3); // Truncated
    }

    #[test]
    fn test_fractional() {
        let half = Fixed64::HALF;
        let one = Fixed64::ONE;

        assert_eq!((half + half), one);
        assert_eq!((half * Fixed64::from_int(4)).int_part(), 2);
    }

    #[test]
    fn test_thresholds() {
        // Verify threshold constants are correctly encoded
        assert!((thresholds::IHSAN_MIN.to_f64() - 0.99).abs() < 0.0001);
        assert!((thresholds::SNR_MIN.to_f64() - 1.5).abs() < 0.0001);
        assert!((thresholds::CONSENSUS_QUORUM.to_f64() - 0.70).abs() < 0.0001);
    }

    #[test]
    fn test_serde_round_trip() {
        let value = Fixed64::from_f64(0.75);
        let json = serde_json::to_string(&value).expect("serialize Fixed64");
        let parsed: f64 = serde_json::from_str(&json).expect("parse f64");
        assert!((parsed - 0.75).abs() < 1e-9);

        let round: Fixed64 = serde_json::from_str(&json).expect("deserialize Fixed64");
        assert!((round.to_f64() - 0.75).abs() < 1e-9);

        let int_round: Fixed64 = serde_json::from_str("2").expect("deserialize integer");
        assert_eq!(int_round.int_part(), 2);
    }
}
