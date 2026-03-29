import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import tensorflow as tf

# CONSTANTS
#N = 64
max_delay = 0
NUM_SC = 612
NUM_SLOTS = 14
ALPHA = 1/np.sqrt(10) # Normalization factor for QAM Mapping
NUM_DMRS_SYMBOLS = 816
# Using FR1 so setting subcarrier spacing to 15kHz
subcarrier_spacing = 15e3
center_frequency = 5e6
# Max delay = 3e-6
# So, symbol duration = 1/15e3 = 66.67e-6 seconds
# T_slot = 10^-3 / 2^u = 10^-3 / 2^0 = 1 ms
# CP duration, Tcp = (T_slot - 14*Tsym)/14 = 4.762 us > Max. Delay
# N - point FFT = 612
# Sampling period, Ts = 1/ f*N = 1/*15e3 *612) = 0.1089 us
# CP length = Tcp / Ts = 43.715
CP_LENGTH = 44
NUM_RESOURCE_GRID_SYMBOLS = NUM_SC * NUM_SLOTS
NUM_DATA_SYMBOLS = NUM_RESOURCE_GRID_SYMBOLS - NUM_DMRS_SYMBOLS
SYMBOL_LENGTH = NUM_SC + CP_LENGTH
SIGNAL_LEN =  SYMBOL_LENGTH * NUM_SLOTS
#snr_dB = 14
DMRS_CYCLE = np.array([(-1 + 1j), 3 + 1j, 1 - 1j, 1 + 3j]) *ALPHA
DMRS_INDICES = np.linspace(10, 8567, 816, dtype=int) # CHECK POINT 1!!!!!!!!!!
DMRS_POSITIONS = np.array([(i % NUM_SC,  i // NUM_SC) for i in DMRS_INDICES])
RESOURCE_GRID = np.zeros((NUM_SC, NUM_SLOTS), dtype=complex)

# sc_idx = subcarrier index, sym_idx = OFDM symbol index
for i, (sc_idx, sym_idx) in enumerate(DMRS_POSITIONS):
    RESOURCE_GRID[sc_idx, sym_idx] = DMRS_CYCLE[i % len(DMRS_CYCLE)]

DMRS_MASK = np.zeros((NUM_SC, NUM_SLOTS), dtype=bool)
for sc_idx, sym_idx in DMRS_POSITIONS:
    DMRS_MASK[sc_idx, sym_idx] = True

DATA_POSITIONS = []
for sym_idx in range(NUM_SLOTS):
  for sc_idx in range(NUM_SC):
    if not DMRS_MASK[sc_idx, sym_idx]:
      DATA_POSITIONS.append((sc_idx, sym_idx))
DATA_POSITIONS = np.array(DATA_POSITIONS)
PATH_DATAS = np.array([(0.04570882, 0),(1.0, 1.1457e-07),(0.60255959, 1.2075e-07),(0.39810717, 1.7604e-07),
(0.25118864, 1.383e-07), (0.15135612, 1.6125e-07), (0.1023293, 2.0124e-07), (0.08912509, 1.725e-07),
(0.17782794, 2.2854e-07),(0.02570396, 4.6125e-07),(0.21877616, 5.6934e-07),(0.02137962, 6.6726e-07),
(0.05754399, 6.5154e-07),(0.03019952, 7.4826e-07),(0.08317638, 7.5357e-07),(0.07413102, 9.1746e-07),
(0.05370318, 1.2243e-06),(0.02398833, 1.3373e-06),(0.01479108, 1.3709e-06),(0.0128825, 1.439e-06),
(0.02187762, 1.502e-06), (0.01023293, 1.5913e-06), (0.00107152, 2.8976e-06)])

NUM_TAPS = len(PATH_DATAS)

# Array of subcarrier frequencies - center frequency = 5 MHz, subcarrier spacing = 15 kHz
F = np.arange(-NUM_SC//2, NUM_SC//2) * subcarrier_spacing
FR = F + center_frequency

gains = np.array([g for g, _ in PATH_DATAS])
gains /= np.sqrt(np.sum(gains**2))

PATH_DATAS_NORM = [(g, d) for g, (_, d) in zip(gains, PATH_DATAS)]
gains = np.array([g for g, _ in PATH_DATAS_NORM])
pdp_energy = np.sum(gains**2)

print("PDP energy =", pdp_energy)

# Generate Data and modulate

qam_map = {
    (0,0,0,0): ALPHA* (1 + 1j),
    (0,0,0,1): ALPHA* (1 + 3j),
    (0,0,1,1): ALPHA* (1 - 3j),
    (0,0,1,0): ALPHA* (1 - 1j),

    (0,1,0,0): ALPHA* (3 + 1j),
    (0,1,0,1): ALPHA* (3 + 3j),
    (0,1,1,1): ALPHA* (3 - 3j),
    (0,1,1,0): ALPHA* (3 - 1j),

    (1,1,0,0): ALPHA*(-3 + 1j),
    (1,1,0,1): ALPHA*(-3 + 3j),
    (1,1,1,1): ALPHA*(-3 - 3j),
    (1,1,1,0): ALPHA*(-3 - 1j),

    (1,0,0,0): ALPHA*(-1 + 1j),
    (1,0,0,1): ALPHA*(-1 + 3j),
    (1,0,1,1): ALPHA*(-1 - 3j),
    (1,0,1,0): ALPHA*(-1 - 1j),
}

qam_demap = {v: k for k, v in qam_map.items()} # CHECK POINT 2!!!!!!!!

def qam_mapping(bit_stream):
  L = len(bit_stream)
  bit_grps = [tuple(bit_stream[i:i+4]) for i in range(0, L, 4)]
  qam_symbols = np.array([qam_map[grp] for grp in bit_grps])
  return qam_symbols
  
  def random_bits(length):
      return np.random.randint(0,2,length)

# currently this will only ever return a 612 x 14 resource grid
def bit_stream_to_resource_grid(bit_stream):
  data_symbols = qam_mapping(bit_stream)
  resource_grid = np.copy(RESOURCE_GRID)

  count = 0  # for verification
  for n, (sc_idx, sym_idx) in enumerate(DATA_POSITIONS):
    count += 1
    resource_grid[sc_idx, sym_idx] = data_symbols[n]
  return data_symbols,resource_grid

def time_domain_symbols(resource_grid): #change the list name in OFDM Transmitter
  time_domain_symbols = []
  for sym_idx in range(NUM_SLOTS):
      frequency_domain_symbol = resource_grid[:, sym_idx]  # 612 subcarriers
      time_domain_symbol = np.fft.ifft(frequency_domain_symbol, 612)
      time_domain_symbols.append(time_domain_symbol)
  return time_domain_symbols

def apply_cyclic_prefix(time_domain_symbols):
    symbols_with_cp = []
    for time_symbol in time_domain_symbols:
        symbol_with_cp = np.concatenate([time_symbol[-CP_LENGTH:], time_symbol])
        symbols_with_cp.append(symbol_with_cp)
    return np.concatenate(symbols_with_cp)
  
def apply_multipath(path_datas, t_signal):
    Fs = NUM_SC * subcarrier_spacing
    delays_samples = [int(round(delay * Fs)) for _, delay in path_datas]
    h = np.zeros(max(delays_samples) + 1, dtype=complex)
    nlos_gains = np.array([p[0] for p in path_datas])
    for i, n in enumerate(delays_samples):
        h[n] += (nlos_gains[i] / np.sqrt(2)) * (np.random.randn() + 1j*np.random.randn())
    r_signal = np.convolve(t_signal, h, mode='full')[:len(t_signal)]
    H = np.fft.fft(h, NUM_SC)
    return r_signal, H
    
def preprocessing(r_signal):
  received_symbols = []

  for i in range(NUM_SLOTS):
      start_idx = i * SYMBOL_LENGTH
      end_idx = start_idx + SYMBOL_LENGTH

      if end_idx <= len(r_signal):
          symbol_with_cp = r_signal[start_idx:end_idx]
          symbol_no_cp = symbol_with_cp[CP_LENGTH:]
          freq_domain_symbol = np.fft.fft(symbol_no_cp, NUM_SC)
          received_symbols.append(freq_domain_symbol)
  return np.array(received_symbols)

# Applying method of least squares and performing linear interpolation
def LS_strategy(r_signal, *params):
  received_pilots = [(i,value) for i, value in enumerate(r_signal) if i in DMRS_INDICES]
  #print("No. of received pilots:", len(received_pilots))
  channel_response = np.zeros((NUM_SC, NUM_SLOTS), dtype=complex)
  for (idx, val) in received_pilots:
    channel_response[idx % NUM_SC, idx // NUM_SC] = val / RESOURCE_GRID[idx % NUM_SC, idx // NUM_SC]
  return channel_response

def lin_interpolation(x_points, y_points, X):
  results = []
  for x in X:
    if x < x_points[0]:
      results.append (y_points[0])
    elif x > x_points[-1]:
      results.append(y_points[-1])
    else:
      for i in range (len(x_points)-1):
        x1, x2 = x_points[i], x_points[i+1]
        y1, y2 = y_points[i], y_points[i+1]
        if x1 <= x <= x2:
          t = (x - x1) / (x2 - x1)
          results.append(y1 + t * (y2 - y1))
          break
  return np.array(results, dtype = complex)

def channel_interpolation(dmrs_estimate_grid):
  H_est = np.zeros((NUM_SC, NUM_SLOTS), dtype=complex)
  # Interpolation across frequency
  for sym in range(NUM_SLOTS):
    pilot_sc = DMRS_POSITIONS[DMRS_POSITIONS[:,1] == sym, 0]
    pilot_vals = dmrs_estimate_grid[pilot_sc, sym]
    if len(pilot_sc) == 0:
      continue  # no pilots in this symbol
    H_est[:, sym] = lin_interpolation(pilot_sc, pilot_vals, np.arange(NUM_SC))
  return H_est
  
def compute_R_HH():
  tap_powers = PATH_DATAS[:,0]
  tap_delays = PATH_DATAS[:,1]
  R_HH = np.zeros((NUM_SC, NUM_SC), dtype=complex)
  ks = np.arange (NUM_SC)
  for k in range(NUM_SC):
    for m in range(NUM_SC):
      d = k - m
      R_HH[k, m] = np.sum(tap_powers * np.exp(-1j *2*np.pi*d*subcarrier_spacing*tap_delays))
  return R_HH

def compute_R_HY(R_HH_allp, X_p):
    return R_HH_allp @ np.conjugate(X_p.T)

def compute_R_YY(noise_var, X_p, R_HH_pp):
  R_YY = X_p @ R_HH_pp @ np.conjugate(X_p.T) + noise_var* np.eye(816)
  return R_YY

def LMMSE(R_HY, R_YY, r_signal):
  r_dmrs = r_signal[DMRS_INDICES].reshape(-1, 1)
  return R_HY @ np.linalg.inv(R_YY) @ r_dmrs
  
# AWGN Channel
def add_AWGN(signal, snr_dB):
  signal_power = np.mean(np.abs(signal)**2)
  snr = 10 ** (snr_dB/10)
  noise_power = signal_power/snr
  noise_real = np.random.normal(0, np.sqrt(noise_power/2), len(signal))
  noise_imag = np.random.normal(0, np.sqrt(noise_power/2), len(signal))
  noise = noise_real + 1j * noise_imag
  received_signal = signal + noise

  return received_signal, noise
  
def square_distance(cmplx1, cmplx2):
  return (cmplx1.real - cmplx2.real)**2 + (cmplx1.imag - cmplx2.imag)**2

# turns complex number into bit pattern corrsesponding to closest QAM symbol
def ML_Decoder(qam_symbol):
  qam_symbols = list(qam_demap.keys())
  min_index = np.argmin([square_distance(qam_symbol, sym) for sym in qam_symbols])
  closest_symbol = qam_symbols[min_index]
  return qam_demap[closest_symbol]

# Adds pad_length elements from end of qam_symbols to beginnging
def cyclic_prefix(qam_symbols, pad_length):
  return qam_symbols[-pad_length:] + qam_symbols

# Maps list of complex numbers to a list of bits
def qam_demapping(qam_symbols):
  decoded_bits = []
  for symbol in qam_symbols:
    bits = ML_Decoder(symbol)
    decoded_bits.extend(bits)
  return np.array(decoded_bits)

def calculate_mse(original_symbols, received_symbols):
    """Calculate Mean Square Error between symbol constellations"""
    if len(original_symbols) != len(received_symbols):
        min_len = min(len(original_symbols), len(received_symbols))
        original_symbols = original_symbols[:min_len]
        received_symbols = received_symbols[:min_len]

    mse = np.mean(np.abs(np.array(original_symbols) - np.array(received_symbols))**2)
    return mse
def calculate_ber(original_bits, decoded_bits):
    """Calculate Bit Error Rate"""
    if len(original_bits) != len(decoded_bits):
        min_len = min(len(original_bits), len(decoded_bits))
        original_bits = original_bits[:min_len]
        decoded_bits = decoded_bits[:min_len]

    errors = np.sum(original_bits != decoded_bits)
    ber = errors / len(original_bits)
    return ber, errors
    
def signal_to_grid(signal):
  received_symbols = []

  for i in range(NUM_SLOTS):
      start_idx = i * SYMBOL_LENGTH
      end_idx = start_idx + SYMBOL_LENGTH

      if end_idx <= len(signal):
          symbol_with_cp = signal[start_idx:end_idx]
          symbol_no_cp = symbol_with_cp[CP_LENGTH:]
          freq_domain_symbol = np.fft.fft(symbol_no_cp, NUM_SC)
          received_symbols.append(freq_domain_symbol)

  received_grid = np.zeros((NUM_SC, NUM_SLOTS), dtype=complex)
  for sym_idx, freq_symbol in enumerate(received_symbols):
      received_grid[:, sym_idx] = freq_symbol
  return received_grid

def grid_to_data_symbols(grid):
  received_data_symbols = []
  for sc_idx, sym_idx in DATA_POSITIONS:
      received_data_symbols.append(grid[sc_idx, sym_idx])
  # print(f"Extracted {len(received_data_symbols)} received data symbols")
  return received_data_symbols
  
# ZF equalizer and MMSE Equalizer
def zf_equalizer(r_signal, channel_response, epsilon=1e-10):
  return r_signal / np.where(np.abs(channel_response) > epsilon, channel_response, epsilon)

def mmse_equalizer(r_signal, channel_response, noise_var, symbol_power=1.0):
    H_conj = np.conj(channel_response)
    denom = np.abs(channel_response)**2 + noise_var/symbol_power
    return (H_conj / denom) * r_signal
    
def transmit(signal, SNR_DB):
  r_signal, H_true = apply_multipath(PATH_DATAS_NORM, signal)
  received_signal_awgn, noise_multipath = add_AWGN(r_signal, SNR_DB)
  return received_signal_awgn, H_true, noise_multipath
  
def preprocessing_received_signal(received_signal):
  pre_processed_ofdm_symbols = preprocessing(received_signal)
  flattened_preprocessed_symbols = np.concatenate(pre_processed_ofdm_symbols)
  return flattened_preprocessed_symbols, LS_strategy(flattened_preprocessed_symbols)
