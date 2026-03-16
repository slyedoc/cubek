use crate::components::resource::CubeDimResource;
use crate::components::tile::TileMatmulFamily;
use crate::components::tile::mma::config::MmaMatmulConfig;
use crate::components::tile::scaled_mma::ScaledMmaMatmul;
use crate::definition::{MatmulAvailabilityError, MatmulSetupError, MatmulVectorSizes};
use crate::definition::{MatmulElems, TilingBlueprint};
use cubecl::ir::{DeviceProperties, StorageType};
use cubecl::features::MmaConfig;
use cubecl::prelude::*;
use cubek_std::tile::{Strided, TileKind};
use cubek_std::tile::mma::{MmaFragmentReader, MmaStageReader};
use cubek_std::{InvalidConfigError, TileSize};

impl<LhsTile: TileKind, RhsTile: TileKind, AccTile: TileKind> TileMatmulFamily
    for ScaledMmaMatmul<LhsTile, RhsTile, AccTile>
where
    MmaStageReader<LhsTile>: MmaFragmentReader<TileKind = LhsTile>,
    MmaStageReader<RhsTile>: MmaFragmentReader<TileKind = RhsTile>,
    MmaStageReader<AccTile>: MmaFragmentReader<TileKind = AccTile>,
{
    type Config = MmaMatmulConfig;
    type Matmul<L: Numeric, R: Numeric, A: Numeric> =
        ScaledMmaMatmul<LhsTile, RhsTile, AccTile>;
    type LhsTile = LhsTile;
    type RhsTile = RhsTile;
    type AccTile = AccTile;
    type OutTile = Strided;

    fn requires_accelerator() -> bool {
        true
    }

    fn can_cast_stage_element() -> bool {
        false
    }

    fn cubedim_resource() -> Result<CubeDimResource, InvalidConfigError> {
        Ok(CubeDimResource::Planes(1))
    }

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<MmaMatmulConfig, MatmulSetupError> {
        use cubek_std::tile::mma::MmaIOConfig;
        Ok(MmaMatmulConfig {
            shared: crate::components::tile::SharedTileConfig {
                tile_size: blueprint.tiling_scheme.tile_size,
                plane_dim: blueprint.plane_dim,
                swizzle_modes: blueprint.swizzle_modes,
            },
            mma_io_config: MmaIOConfig::new(
                device_props,
                dtypes.lhs_stage,
                dtypes.rhs_stage,
                dtypes.acc_stage,
            ),
        })
    }

    fn should_swizzle<R: Runtime>(_client: &ComputeClient<R>) -> bool {
        false
    }

    fn is_supported<R: Runtime>(client: &ComputeClient<R>, config: MmaConfig) -> bool {
        // Check for scaled MMA support (not regular MMA/CMMA)
        // The config here carries the types — we check if the hardware supports
        // block-scaled MMA with these operand types.
        client.properties().features.scaled_mma.iter().any(|c| {
            c.a_type == config.a_type
                && c.b_type == config.b_type
                && c.cd_type == config.cd_type
                && c.m == config.m
                && c.n == config.n
                && c.k == config.k
        })
    }

    fn supported_sizes<R: Runtime>(
        client: &ComputeClient<R>,
        lhs_ty: StorageType,
        rhs_ty: StorageType,
        acc_ty: StorageType,
    ) -> Vec<TileSize> {
        client
            .properties()
            .features
            .scaled_mma
            .iter()
            .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
            .map(|it| (it.m, it.n, it.k).into())
            .collect()
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &TilingBlueprint,
        dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        let lhs = dtypes.lhs_register;
        let rhs = dtypes.rhs_register;
        let acc = dtypes.acc_register;
        let size = blueprint.tiling_scheme.tile_size;

        let supported = client.properties().features.scaled_mma.iter().any(|c| {
            c.a_type == lhs
                && c.b_type == rhs
                && c.cd_type == acc
                && c.m == size.m()
                && c.n == size.n()
                && c.k == size.k()
        });

        if !supported {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs,
                    rhs,
                    output: acc,
                    size: Some(TileSize::new(size.m(), size.n(), size.k())),
                },
            ));
        }

        Ok(())
    }
}
