import tensorflow as tf
import numpy as np
import h5py
from tensorflow_model_optimization.quantization.keras import vitis_quantize

print("\nLoading Model\n")
model = tf.keras.models.load_model("cnn_ofdm_estimator_vitis_relu.h5", compile=False)
print("\nModel Loaded Successfully.\n")

def calibration_generator(h5_path, max_samples=200):
    with h5py.File(h5_path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("model_input_"))
        for i, k in enumerate(keys):
            if i >= max_samples:
                break
            data = f[k][:]
            yield np.expand_dims(data, axis=0)

print("\nType of model:", type(model))
print(isinstance(model, tf.keras.Model))
print("\nCreating Quantizer\n")
for layer in model.layers:
    layer._name = layer.name.strip()

quantizer = vitis_quantize.VitisQuantizer(model)

print("\nStarting Quantization\n")
quantized_model = quantizer.quantize_model(
    calib_dataset=calibration_generator("calibration_data.h5", 200),
)

print("Quantization Complete. Saving model...")
quantized_model.save("quantized_relu.h5")
print("Saved as quantized_relu.h5")

