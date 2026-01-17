// src/sape/mod.rs
pub mod base;
pub mod elevator;
pub mod graph;
pub mod harness;
pub mod ihsan;
pub mod pattern_compiler;
pub mod tension;

pub use base::{get_sape, ProbeDimension, ProbeResult, SAPEEngine, SnrTier, TieredProbeResult};
pub use elevator::AbstractionElevator;
pub use graph::{EdgeType, NodeType, ReasoningGraph};
pub use harness::SymbolicHarness;
pub use pattern_compiler::{
    OptimizationLevel, Pattern, PatternCompiler, PatternError, PatternStats,
};
pub use tension::TensionStudio;
