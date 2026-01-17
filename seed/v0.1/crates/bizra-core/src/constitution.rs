use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct InvariantId(pub &'static str);

/// The constitution is intentionally small. Everything else is *derived*.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constitution {
    pub ihsan_threshold: f32,
    pub invariants: Vec<InvariantId>,
}

impl Constitution {
    pub fn seed() -> Self {
        Self {
            ihsan_threshold: 0.85,
            invariants: vec![
                InvariantId("L0-01"),
                InvariantId("L0-02"),
                InvariantId("L0-03"),
                InvariantId("L0-04"),
                InvariantId("L0-05"),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constitution_has_threshold() {
        let c = Constitution::seed();
        assert!(c.ihsan_threshold >= 0.85);
    }
}
