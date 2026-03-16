#![allow(missing_docs)] // pub cube modules

use cubecl::prelude::*;
use cubecl::{
    calculate_cube_count_elemwise,
    ir::{ElemType, FloatKind, IntKind, StorageType},
};
use cubecl::{features::TypeUsage, tensor_vector_size_parallel};

use crate::{
    layout::{ScalesView, scales_view},
    scheme::{QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue},
};
use cubecl::std::tensor::{
    View,
    layout::linear::{LinearView, linear_view},
};

/// Dequantize a vector of values into floating-point values using the provided scale.
#[cube]
pub fn dequantize_symmetric<F: Float, FS: CubePrimitive, N: Size>(
    value: Vector<F, N>,
    scale: FS,
) -> Vector<F, N> {
    // x = scale * x_q
    Vector::cast_from(scale) * value
}

/// Dequantize the value at a specified position using the provided quantization scheme.
///
/// Returns a vector of floating-point values. The number of values in the vector depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_values<
    F: Float,
    NF: Size,
    FS: CubePrimitive,
    QI: Int,
    NQ: Size,
>(
    position: usize,
    values: &View<Vector<QI, NQ>, usize>,
    scales: &View<FS, usize>,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    dequantize_symmetric_packed_value_at::<F, NF, FS, QI, NQ>(
        position,
        values[position],
        scales,
        scheme,
    )
}

/// Dequantize a single value using the scale at the specified position.
///
/// Returns a vector of floating-point values. The number of values in the vector depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_value_at<
    F: Float,
    NF: Size,
    FS: CubePrimitive,
    QI: Int,
    NQ: Size,
>(
    position: usize,
    values: Vector<QI, NQ>,
    scales: &View<FS, usize>,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    dequantize_symmetric_packed_value::<F, NF, FS, QI, NQ>(values, scales, position, scheme)
}

/// Dequantize a single packed value using the scale provided.
///
/// Returns a vector of floating-point values. The number of values in the vector depends on the number of packed
/// values in the stored quantization type.
#[cube]
pub fn dequantize_symmetric_packed_value<
    F: Float,
    NF: Size,
    FS: CubePrimitive,
    QS: Int,
    NQ: Size,
>(
    values: Vector<QS, NQ>,
    scales: &View<FS, usize>,
    position: usize,
    #[comptime] scheme: QuantScheme,
) -> Array<Vector<F, NF>> {
    let vector_size_values = values.vector_size();
    let num_quants = scheme.num_quants();
    let mut tmp = Array::new(vector_size_values);

    #[unroll]
    for i in 0..vector_size_values {
        let floats = unpack_q::<F, NF, QS>(values[i], scheme.value, scheme.store);
        let scale = scales[(position * vector_size_values) + i * num_quants];
        let values = dequantize_symmetric::<F, FS, NF>(floats, scale);
        tmp[i] = values;
    }

    tmp
}

/// Unpack a quantized integer into a vector of floating-point values, according to the specified quantization input type.
///
/// This handles types where multiple quantized values are packed into a single integer (the stored quantization type).
#[allow(clippy::explicit_counter_loop)]
#[cube]
fn unpack_q<F: Float, NF: Size, QS: Int>(
    value: QS,
    #[comptime] quant: QuantValue,
    #[comptime] store: QuantStore,
) -> Vector<F, NF> {
    let size_quant = quant.size_bits();
    let size_store = store.size_bits(&quant);
    let num_quant = size_store / size_quant;

    let mut output = Vector::empty();

    let mask = QS::from_int((1 << size_quant) - 1);
    let sign_bit = QS::from_int(1 << (size_quant - 1));
    let two_pow_n = 1 << size_quant;

    #[unroll]
    for position in 0..num_quant {
        let offset = QS::cast_from(position * size_quant);
        let raw = (value >> offset) & mask;

        // Branchless two's complement conversion
        // If raw >= 2^(n-1), then result = raw - 2^n
        let raw_i32 = i32::cast_from(raw);
        let is_negative = i32::cast_from(raw >= sign_bit); // 1 if negative, 0 if positive
        let signed_value = raw_i32 - (is_negative * two_pow_n);

        output[position] = F::cast_from(signed_value);
    }

    output
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_packed_kernel<F: Float, NF: Size, FS: Numeric, NQ: Size>(
    input: &LinearView<Vector<u32, NQ>>,
    scales: &ScalesView<FS>,
    output: &mut LinearView<Vector<F, NF>, ReadWrite>,
    #[comptime] scheme: QuantScheme,
    #[define(F, FS)] _dtypes: [StorageType; 2],
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let vector_size_in = input.vector_size();
    let vector_size_out = output.vector_size();

    comptime! {
        assert_eq!(vector_size_out, scheme.num_quants());
    }

    let values = input[ABSOLUTE_POS];
    let packed_pos = ABSOLUTE_POS * scheme.num_quants();

    let out =
        dequantize_symmetric_packed_value::<F, NF, FS, u32, NQ>(values, scales, packed_pos, scheme);

    #[unroll]
    for i in 0..vector_size_in {
        output[ABSOLUTE_POS * vector_size_in + i] = out[i];
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn dequantize_symmetric_native_kernel<F: Float, NF: Size, FS: Numeric, Q: Numeric, NQ: Size>(
    input: &LinearView<Vector<Q, NQ>>,
    scale: &ScalesView<FS>,
    output: &mut LinearView<Vector<F, NF>, ReadWrite>,
    #[define(F, FS, Q)] _dtypes: [StorageType; 3],
) {
    if !input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let native_packing = Q::packing_factor();
    let scale = scale[ABSOLUTE_POS * input.vector_size() * native_packing];

    if native_packing == 1 {
        output[ABSOLUTE_POS] =
            dequantize_symmetric::<F, FS, NF>(Vector::cast_from(input[ABSOLUTE_POS]), scale);
    } else {
        // PackedNative (e.g. e2m1x2): NQ=1 packed element, NF=packing floats.
        // Vector::cast_from handles the packed→unpacked expansion via special_cast.
        output[ABSOLUTE_POS] =
            dequantize_symmetric::<F, FS, NF>(Vector::cast_from(input[ABSOLUTE_POS]), scale);
    }
}

#[allow(clippy::result_large_err)]
/// Convert the tensor back to a higher precision data type.
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    values: TensorBinding<R>,
    output: TensorBinding<R>,
    params: TensorBinding<R>,
    scheme: &QuantScheme,
    input_dtype: StorageType,
) -> Result<(), LaunchError> {
    let dtype_scale: StorageType = ElemType::from_quant_param(scheme.param).into();

    match scheme {
        QuantScheme {
            store: QuantStore::PackedU32(_),
            ..
        } => dequantize_packed(
            client,
            values,
            *scheme,
            params,
            output,
            input_dtype,
            dtype_scale,
        ),
        QuantScheme {
            value: QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2,
            store: QuantStore::Native,
            ..
        }
        | QuantScheme {
            value: QuantValue::E2M1,
            store: QuantStore::PackedNative(_),
            ..
        } => {
            if !i8::supported_uses(client).contains(TypeUsage::Conversion) {
                panic!(
                    "{:?} is not supported for native quantization",
                    scheme.value
                );
            }

            dequantize_native(
                client,
                values,
                *scheme,
                params,
                output,
                input_dtype,
                dtype_scale,
            )
        }
        QuantScheme {
            store: QuantStore::Native | QuantStore::PackedNative(_),
            value,
            ..
        } => {
            panic!("{value:?} is not supported for native quantization");
        }
    }
}

fn dequantize_packed<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    scheme: QuantScheme,
    scale: TensorBinding<R>,
    output: TensorBinding<R>,
    input_dtype: StorageType,
    scale_dtype: StorageType,
) -> Result<(), LaunchError> {
    let num_elems_input: usize = input.shape.iter().product();

    let mut vector_size_in = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(input_dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );
    let num_quants = scheme.num_quants();
    let vector_size_out = num_quants;
    let rank = output.shape.len();

    if !output.shape[rank - 1].is_multiple_of(vector_size_out) {
        vector_size_in = 1;
    }

    let num_elems = num_elems_input / vector_size_in as usize;
    let cube_dim = CubeDim::new(client, num_elems);
    let cube_count = calculate_cube_count_elemwise(client, num_elems, cube_dim);
    let address_type = input
        .required_address_type(size_of::<u32>())
        .max(scale.required_address_type(scale_dtype.size()))
        .max(output.required_address_type(input_dtype.size()));

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            store: QuantStore::PackedU32(_),
            mode: QuantMode::Symmetric,
            ..
        } => unsafe {
            dequantize_symmetric_packed_kernel::launch_unchecked(
                client,
                cube_count,
                cube_dim,
                address_type,
                vector_size_out,
                vector_size_in,
                linear_view(client, input.clone(), vector_size_in),
                scales_view(client, input, scale, 1, &scheme),
                linear_view(client, output, vector_size_out),
                scheme,
                [input_dtype, scale_dtype],
            )
        },
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    };

    Ok(())
}

fn dequantize_native<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    mut scheme: QuantScheme,
    scale: TensorBinding<R>,
    output: TensorBinding<R>,
    input_dtype: StorageType,
    scale_dtype: StorageType,
) -> Result<(), LaunchError> {
    // For PackedNative on non-innermost dim, re-pack to innermost first
    let input = match scheme.store {
        QuantStore::PackedNative(packed_dim) if packed_dim != 0 => {
            // Use raw u8 dtype for the copy, matching what the matmul launch does
            let raw_dtype = u8::as_type_native_unchecked().storage_type();
            let mut repacked = cubecl::std::tensor::into_contiguous_packed(
                client,
                input,
                packed_dim,
                &output.shape, // unpacked shape
                scheme.num_quants(),
                raw_dtype,
            );
            scheme = scheme.with_store(QuantStore::PackedNative(0));
            // Restore the actual packed dtype for the dequantize kernel
            repacked.dtype = StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2);
            repacked.binding()
        }
        _ => input,
    };

    let num_elems: usize = input.shape.iter().product();
    let quant_type = ElemType::from_quant_value(scheme.value);
    let packing = scheme.num_quants();
    let vector_size = if packing > 1 {
        1 // PackedNative: scalar reads, kernel handles unpacking
    } else {
        tensor_vector_size_parallel(
            client
                .io_optimized_vector_sizes(input_dtype.size())
                .filter(|&v| v <= quant_type.max_vector_size() as usize),
            &input.shape,
            &input.strides,
            input.shape.len() - 1,
        )
    };
    let working_units = num_elems / vector_size;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            value,
            store: QuantStore::Native | QuantStore::PackedNative(_),
            ..
        } => {
            let quant_dtype: StorageType = match (value, &scheme.store) {
                (QuantValue::Q8F | QuantValue::Q8S, _) => ElemType::Int(IntKind::I8).into(),
                (QuantValue::E4M3, _) => ElemType::Float(FloatKind::E4M3).into(),
                (QuantValue::E5M2, _) => ElemType::Float(FloatKind::E5M2).into(),
                // PackedNative E2M1 → e2m1x2 (packed pair), not bare e2m1
                (QuantValue::E2M1, QuantStore::PackedNative(_)) => {
                    StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2)
                }
                (QuantValue::E2M1, _) => ElemType::Float(FloatKind::E2M1).into(),
                (other, _) => panic!("Unsupported quantization value {other:?}"),
            };

            let address_type = input
                .required_address_type(quant_dtype.size())
                .max(scale.required_address_type(scale_dtype.size()))
                .max(output.required_address_type(input_dtype.size()));

            // For PackedNative: input vector_size=1 (one packed element),
            // output vector_size=packing (multiple floats per packed element)
            let out_vector_size = if packing > 1 { packing } else { vector_size };

            unsafe {
                dequantize_symmetric_native_kernel::launch_unchecked(
                    client,
                    cube_count,
                    cube_dim,
                    address_type,
                    out_vector_size, // NF: output float vector size
                    vector_size,     // NQ: input quant vector size
                    linear_view(client, input.clone(), vector_size),
                    scales_view(client, input, scale, 1, &scheme),
                    linear_view(client, output, out_vector_size),
                    [input_dtype, scale_dtype, quant_dtype.into()],
                )
            }
        }
        QuantScheme { .. } => panic!("Unsupported quantization scheme {scheme:?}"),
    };

    Ok(())
}
