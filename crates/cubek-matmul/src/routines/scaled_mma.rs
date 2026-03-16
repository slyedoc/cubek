//! Scaled MMA matmul kernel for hardware FP4 (E2M1) on Blackwell (sm_120+).
//!
//! Direct kernel that reads packed e2m1x2 data and ue8m0 scales from global
//! memory and executes hardware scaled MMA instructions.
//!
//! This is a standalone kernel that bypasses the staged pipeline.
//! For production use, it should be integrated into the tiling pipeline
//! with shared memory staging for better performance.

// TODO: Implement the actual scaled MMA kernel.
//
// The kernel needs to:
// 1. Tile M/N/K dimensions across cubes (one output tile per cube)
// 2. For each K tile:
//    a. Load e2m1x2 data into MMA registers via MmaDefinition::position_of_nth
//    b. Load ue8m0 scales via MmaDefinition::scales_index
//    c. Call MmaDefinition::execute_scaled
// 3. Write accumulated f32 output
//
// The cubecl test at cubecl-core/src/runtime_tests/cmma.rs (kernel_scaled)
// shows the single-tile pattern. This kernel extends it to full matmul
// with M/N/K tiling.
//
// Challenges:
// - Size generics (NA, NB, NC, NS) must be defined at the function level,
//   not inside the cube function. Use the same pattern as mma/matmul.rs
//   with define_size! at module scope.
// - Scale indexing depends on the block scaling layout which may not
//   align with the MMA tile K dimension.
// - B matrix layout: MMA expects col-major for B, but the packed data
//   may be row-major. Need layout transformation.
//
// This will be implemented in a follow-up.
