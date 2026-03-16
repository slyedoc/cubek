/// Naive non-cooperative matmul without tiling that can be very fast on small matrices.
pub mod naive;
/// Hardware scaled MMA matmul for FP4 (E2M1) on Blackwell.
pub mod scaled_mma;

pub mod double_buffering;
pub mod double_unit;
pub mod interleaved;
pub mod ordered_double_buffering;
pub mod simple;
pub mod simple_unit;
pub mod specialized;
pub mod vecmat;

mod base;
mod selector;

pub use base::*;
pub use selector::*;
