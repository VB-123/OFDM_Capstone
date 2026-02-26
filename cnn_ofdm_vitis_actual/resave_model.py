import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

# Path where your weights are stored
weights_dir = "weights_npy"

# --- Build model using Functional API (ensures correct input channel inference) ---
inp = Input(shape=(612, 14, 2), name='H_LS_input')
x = Conv2D(64, (5, 5), activation='relu', padding='same', name='conv2d')(inp)
x = Conv2D(64, (5, 1), activation='relu', padding='same', name='conv2d_1')(x)
x = Conv2D(64, (5, 1), activation='relu', padding='same', name='conv2d_2')(x)
x = Conv2D(32, (3, 1), activation='relu', padding='same', name='conv2d_3')(x)
out = Conv2D(2, (3,1), activation='linear', padding='same', name='conv2d_4')(x)

model = Model(inputs=inp, outputs=out)
print("Functional model created successfully.\n")
print("Weights loaded from epoch_50.weights.h5")
# --- Assign weights from saved numpy arrays ---
for layer in model.layers:
    # Only assign weights to Conv2D layers
    if isinstance(layer, Conv2D):
        kernel_path = os.path.join(weights_dir, f"{layer.name}_0.npy")
        bias_path   = os.path.join(weights_dir, f"{layer.name}_1.npy")
        kernel = np.load(kernel_path)
        bias   = np.load(bias_path)
        layer.set_weights([kernel, bias])
        print(f"Weights assigned to layer: {layer.name}")

print("\nAll weights loaded successfully.\n")

# --- Optional: compile model (no loss/optimizer needed for inference/quantization) ---
model.compile()
print("Model compiled.\n")

# --- Save as H5 for Vitis AI quantization ---
model.save("cnn_ofdm_estimator_vitis_relu.h5")
print("Model saved as 'cnn_ofdm_estimator_vitis_relu.h5' and ready for Vitis quantization.\n")

