use cubecl::{define_size, prelude::*};
use cubecl_common::ue8m0;
use cubek_std::{
    MatrixLayout,
    tile::{
        Strided, StridedTile, TileKind,
        mma::{MmaFragmentReader, MmaStageReader, MmaStageWriter},
    },
};
use std::marker::PhantomData;

use crate::components::tile::TileMatmul;
use crate::components::tile::mma::config::MmaMatmulConfig;
use cubecl::{cmma::MmaDefinition, ir::MatrixIdent};

/// Tile matmul using hardware scaled MMA for block-scaled sub-byte types (E2M1/FP4).
///
/// Uses `MmaDefinition::execute_scaled` which takes packed e2m1x2 operands
/// and ue8m0 block scaling factors, executing on hardware tensor cores
/// without software dequantization.
///
/// The Fragment types carry data registers (e2m1x2 for A/B, f32 for accumulator)
/// and scale registers (ue8m0) loaded from the scale tensors passed through
/// the global read pipeline.
pub struct ScaledMmaMatmul<
    Lhs: TileKind = Strided,
    Rhs: TileKind = Strided,
    Acc: TileKind = Option<Strided>,
> {
    _ty: PhantomData<(Lhs, Rhs, Acc)>,
}

define_size!(NL);
define_size!(NR);
define_size!(NA);
define_size!(NS);

/// Fragment for data operands (A or B) — carries both data and scale registers.
#[derive(CubeType)]
pub struct ScaledFragment<E: Numeric, N: Size> {
    data: Array<Vector<E, N>>,
    scales: Vector<ue8m0, NS>,
    #[cube(comptime)]
    layout: MatrixLayout,
}

/// Fragment for the accumulator — no scales needed, just f32 registers.
#[derive(CubeType)]
pub struct AccFragment<A: Numeric, N: Size> {
    fragment: Array<Vector<A, N>>,
    #[cube(comptime)]
    layout: MatrixLayout,
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric, LhsTile: TileKind, RhsTile: TileKind, AccTile: TileKind>
    TileMatmul<L, R, A> for ScaledMmaMatmul<LhsTile, RhsTile, AccTile>
where
    MmaStageReader<LhsTile>: MmaFragmentReader<TileKind = LhsTile>,
    MmaStageReader<RhsTile>: MmaFragmentReader<TileKind = RhsTile>,
    MmaStageReader<AccTile>: MmaFragmentReader<TileKind = AccTile>,
{
    type Config = MmaMatmulConfig;

    type LhsFragment = ScaledFragment<L, NL>;
    type RhsFragment = ScaledFragment<R, NR>;
    type AccFragment = AccFragment<A, NA>;

    type LhsTile = LhsTile;
    type RhsTile = RhsTile;
    type AccTile = AccTile;
    type OutTile = Strided;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        out: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        let def = scaled_mma_definition::<L, R, A>(config);
        let out_arr = def.execute_scaled(
            &lhs.data,
            &rhs.data,
            &out.fragment,
            lhs.scales,
            rhs.scales,
        );
        let num_vectors = def.vectors_per_lane(MatrixIdent::Accumulator);

        #[unroll]
        for i in 0..num_vectors {
            out.fragment[i] = out_arr[i];
        }
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        let def = scaled_mma_definition::<L, R, A>(config);
        register_vector_sizes(def);
        let vector_count = def.vectors_per_lane(MatrixIdent::A);

        ScaledFragment::<L, NL> {
            data: Array::new(vector_count),
            scales: Vector::empty(),
            layout,
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        let def = scaled_mma_definition::<L, R, A>(config);
        register_vector_sizes(def);
        let vector_count = def.vectors_per_lane(MatrixIdent::B);

        ScaledFragment::<R, NR> {
            data: Array::new(vector_count),
            scales: Vector::empty(),
            layout,
        }
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        let def = scaled_mma_definition::<L, R, A>(config);
        register_vector_sizes(def);
        let vector_count = def.vectors_per_lane(MatrixIdent::Accumulator);

        AccFragment::<A, NA> {
            fragment: Array::new(vector_count),
            layout,
        }
    }

    fn load_lhs<E: Numeric, N: Size>(
        tile: &LhsTile::Tile<E, N>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
        // Load data registers from the tile (same as MmaMatmul)
        MmaStageReader::<Self::LhsTile>::load_fragment(
            tile,
            &mut lhs.data,
            scaled_mma_definition::<L, R, A>(config),
            MatrixIdent::A,
            lhs.layout,
            config.shared.tile_size,
            config.mma_io_config,
        );
        // TODO: Load scales from the global scale buffer
        // Requires access to the scale tensor, which the tile doesn't provide.
        // This will be addressed by either:
        // 1. Extending the stage to pass scale data alongside tiles
        // 2. Adding a separate load_scales method to the pipeline
        // 3. Loading scales from a global view stored in the config
    }

    fn load_rhs<E: Numeric, N: Size>(
        tile: &RhsTile::Tile<E, N>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageReader::<Self::RhsTile>::load_fragment(
            tile,
            &mut rhs.data,
            scaled_mma_definition::<L, R, A>(config),
            MatrixIdent::B,
            rhs.layout,
            config.shared.tile_size,
            config.mma_io_config,
        );
        // TODO: Load scales (same as lhs)
    }

    fn load_acc<E: Numeric, N: Size>(
        tile: &AccTile::Tile<E, N>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageReader::<Self::AccTile>::load_fragment(
            tile,
            &mut acc.fragment,
            scaled_mma_definition::<L, R, A>(config),
            MatrixIdent::Accumulator,
            acc.layout,
            config.shared.tile_size,
            config.mma_io_config,
        );
    }

    fn write_results<E: Numeric, N: Size>(
        tile: &mut StridedTile<E, N, ReadWrite>,
        out: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageWriter::store_fragment(
            tile,
            &out.fragment,
            scaled_mma_definition::<L, R, A>(config),
            MatrixIdent::Accumulator,
            tile.layout,
            config.shared.tile_size.m(),
            config.mma_io_config,
        );
    }
}

/// Create a scaled MmaDefinition for E2M1 with UE8M0 scales.
///
/// The scales_factor is determined by the block scaling configuration —
/// for FLUX with block_size=32 and tile_k=64, scales_factor=2
/// (one scale per 32 elements along K, 2 scales per tile).
#[cube]
fn scaled_mma_definition<L: Numeric, R: Numeric, A: Numeric>(
    #[comptime] config: MmaMatmulConfig,
) -> MmaDefinition<L, R, A> {
    let size = config.shared.tile_size;
    let m = comptime![size.m() as usize];
    let n = comptime![size.n() as usize];
    let k = comptime![size.k() as usize];
    // TODO: scales_factor should come from the QuantScheme's block_size
    // relative to tile_k. For now hardcode 2 (block_size=32, tile_k=64).
    let scales_factor = comptime![2usize];
    MmaDefinition::new_scaled::<ue8m0>(m, n, k, scales_factor)
}

#[cube]
#[allow(unused_variables)]
fn register_vector_sizes<L: Numeric, R: Numeric, A: Numeric>(
    def: MmaDefinition<L, R, A>,
) {
    let vector_size_a = def.vector_size(MatrixIdent::A);
    let vector_size_b = def.vector_size(MatrixIdent::B);
    let vector_size_acc = def.vector_size(MatrixIdent::Accumulator);
    let scales_vector_size = def.scales_vector_size();
    intrinsic!(|scope| {
        scope.register_size::<NL>(vector_size_a);
        scope.register_size::<NR>(vector_size_b);
        scope.register_size::<NA>(vector_size_acc);
        scope.register_size::<NS>(scales_vector_size);
    });
}
