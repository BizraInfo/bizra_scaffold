//! FFI Panic Airlock - Converts Rust panics to Python exceptions
//!
//! SECURITY: This module prevents Rust panics from unwinding into Python,
//! which would cause undefined behavior and potential security vulnerabilities.
//!
//! REVIEW FIX: Now preserves panic information for debugging while remaining safe.

use pyo3::{exceptions::PyRuntimeError, PyErr, PyResult};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Extract a human-readable message from a panic payload
fn extract_panic_message(panic_info: &Box<dyn Any + Send>) -> String {
    // Try to extract &str
    if let Some(s) = panic_info.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    // Try to extract String
    if let Some(s) = panic_info.downcast_ref::<String>() {
        return s.clone();
    }
    // Fallback for unknown panic types
    "unknown panic payload".to_string()
}

/// Wraps a closure to catch Rust panics and convert them to Python exceptions.
///
/// # Safety
///
/// This function uses `AssertUnwindSafe` which assumes the closure is unwind-safe.
/// All PyO3 functions that may panic should be wrapped with this.
///
/// # Example
///
/// ```ignore
/// pub fn my_pyfunction() -> PyResult<i32> {
///     panic_airlock(|| {
///         // Code that might panic
///         Ok(42)
///     })
/// }
/// ```
pub fn panic_airlock<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic_info| {
        let msg = extract_panic_message(&panic_info);
        tracing::error!(
            target: "bizra::ffi::panic_airlock",
            panic_message = %msg,
            "Rust panic caught at FFI boundary"
        );
        Err(PyErr::new::<PyRuntimeError, _>(format!(
            "panic_airlock: Rust panic safely caught - {}",
            msg
        )))
    })
}
