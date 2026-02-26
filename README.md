# OFDM_Capstone (WORK IN PROGRESS!!!)
Capstone Project - Implementation of CNN Based OFDM Channel Estimation on PYNQ FPGA

For project files - https://drive.google.com/drive/u/5/folders/1GyqpDyrRLxmQimUA7gIT67d00dwxGsi0

To compile model:
``` vai_c_tensorflow2 -m /workspace/cnn_ofdm_vitis_actual/quantized_relu.h5 -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json -o /workspace/cnn_ofdm_vitis_actual/compiled_model/ -n cnn_ofdm_estimator```
