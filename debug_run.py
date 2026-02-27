from ofdm_utils import *
from pynq_dpu import DpuOverlay
import xir
import vart
overlay = DpuOverlay("/usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq_dpu/dpu.bit")
overlay.download()
print("DPU overlay loaded:", overlay.is_loaded)
overlay.load_model("/home/xilinx/jupyter_notebooks/cnn_ofdm_estimator.xmodel")
dpu_runner = overlay.runner

shapeIn = tuple(dpu_runner.get_input_tensors()[0].dims)
shapeOut = tuple(dpu_runner.get_output_tensors()[0].dims)
in_tensor  = dpu_runner.get_input_tensors()[0]
out_tensor = dpu_runner.get_output_tensors()[0]

print("INPUT:")
print(" dims      :", in_tensor.dims)
print(" dtype     :", in_tensor.dtype)
print(" fix_point :", in_tensor.get_attr("fix_point"))

print("\nOUTPUT:")
print(" dims      :", out_tensor.dims)
print(" dtype     :", out_tensor.dtype)
print(" fix_point :", out_tensor.get_attr("fix_point"))
input_data  = [np.empty(shapeIn,  dtype=np.int8)]
output_data = [np.empty(shapeOut, dtype=np.int8)]
# Check input and output scaling
in_fix = dpu_runner.get_input_tensors()[0].get_attr("fix_point")
out_fix = dpu_runner.get_output_tensors()[0].get_attr("fix_point")
INPUT_SCALE  = 2 ** (-in_fix)
OUTPUT_SCALE = 2 ** (-out_fix)
print("in_fix :", in_fix)
print("out_fix:", out_fix)
#print(f"Input needs to be divided by: {2**in_attr}")
#print(f"Output needs to be divided by: {2**out_attr}")

snr = 16
bit_stream = random_bits(NUM_DATA_SYMBOLS*4)  # QAM turns 4 bits into a single data symbol
data_symbols, resource_grid = bit_stream_to_resource_grid(bit_stream)
signal = apply_cyclic_prefix(time_domain_symbols(resource_grid))
Y, H, noise = transmit(signal, snr)
noise_var = np.mean(np.abs(noise)**2)
flattened_preprocessed_symbols, H_estimate = preprocessing_received_signal(Y)
rx_grid = signal_to_grid(Y)
H_split = np.stack((np.real(H_estimate), np.imag(H_estimate)), axis=-1)  # shape: (612,14,2)
H_batched = np.expand_dims(H_split, axis=0)   # shape: (1, 612, 14, 2)
# --- Quantize to INT8 ---
H_int8 = np.round(H_batched / INPUT_SCALE).astype(np.int8)

# --- DPU buffers ---
input_tensor  = dpu_runner.get_input_tensors()[0]
output_tensor = dpu_runner.get_output_tensors()[0]

in_shape  = tuple(input_tensor.dims)
out_shape = tuple(output_tensor.dims)

input_data  = [np.zeros(in_shape, dtype=np.int8)]
output_data = [np.zeros(out_shape, dtype=np.int8)]

input_data[0][...] = H_int8

# --- Execute on DPU ---
job_id = dpu_runner.execute_async(input_data, output_data)
dpu_runner.wait(job_id)
H_pred_int8 = output_data[0]

H_pred_float = H_pred_int8.astype(np.float32) * OUTPUT_SCALE
H_pred_single = np.squeeze(H_pred_float, axis=0)  # (612,14,2)
H_pred_complex = H_pred_single[..., 0] + 1j * H_pred_single[..., 1]

X_predicted = mmse_equalizer(rx_grid, H_pred_complex, noise_var)

received_data_symbols_cnn = grid_to_data_symbols(X_predicted)
mse_cnn_val = calculate_mse(data_symbols, received_data_symbols_cnn)
bits_cnn = qam_demapping(received_data_symbols_cnn)
ber_cnn_val,_ = calculate_ber(bit_stream, bits_cnn)
print("BER:", ber_cnn_val)
print("MSE:", mse_cnn_val)
print("Input int8 range:", H_int8.min(), H_int8.max())
print("Output int8 range:", output_data[0].min(), output_data[0].max())
del dpu_runner