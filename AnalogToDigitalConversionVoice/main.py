# TODO: 1. Grabar un ejemplo de nota de voz
# TODO: 2. Filtrar los intervalos de silencio
# TODO: 3. Muestreo, cuantificación y codificación
# TODO: 4. Transformacion de Fourier
# TODO: 5. Visualizar el Histograma de frecuencias
# TODO: 6. Analizar y Reporte de resultados

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import resample
import sounddevice as sd
from numpy.fft import fft, fftfreq


# ------------------------DEFINICION DE FUNCIONES-----------------------------------------------------------------------
# Funcion para realizar graficos con matplotlib
def plot_signal(data, title, xlabel, ylabel, color):
    fig, ax = plt.subplots()
    ax.plot(data, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    fig.savefig('./assets/results/' + title + '.png')


# Función para detectar y filtrar intervalos de silencio
def filter_silence(audio_data, samplerate, silence_threshold=0.01, min_silence_duration=0.5):
    # Convertir la duración mínima de silencio a muestras
    min_silence_samples = int(min_silence_duration * samplerate)

    # Si el audio tiene múltiples canales, procesar cada canal por separado
    if len(audio_data.shape) > 1:
        filtered_channels = []
        for channel in range(audio_data.shape[1]):
            filtered_channel = filter_silence(audio_data[:, channel], samplerate, silence_threshold, min_silence_duration)
            filtered_channels.append(filtered_channel)
        return np.stack(filtered_channels, axis=-1)

    # Detectar los intervalos de silencio
    silent_intervals = []
    start_idx = None
    for i, sample in enumerate(audio_data):
        if abs(sample) < silence_threshold:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                if i - start_idx >= min_silence_samples:
                    silent_intervals.append((start_idx, i))
                start_idx = None

    # Filtrar los intervalos de silencio
    filtered_data = []
    last_end = 0
    for start, end in silent_intervals:
        filtered_data.extend(audio_data[last_end:start])
        last_end = end
    filtered_data.extend(audio_data[last_end:])

    return np.array(filtered_data)


# Funcion para realizar el re muestreo
def resample_audio(audio_data, original_samplerate, new_samplerate):
    # Calculate the number of samples in the resampled audio
    num_samples = int(len(audio_data) * new_samplerate / original_samplerate)

    # Resample the audio data
    resampled_data = resample(audio_data, num_samples)

    return resampled_data


# Función para realizar la cuantificación
def quantize(audio_data, num_levels):
    # Normalize the audio data to the range [-1, 1]
    audio_data_normalized = 2 * (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data)) - 1

    # Quantize the normalized data
    quantized_data = np.round((audio_data_normalized + 1) * (num_levels - 1) / 2)

    return quantized_data


# Función para realizar la codificación
def encode(quantized_data):
    # Ensure quantized_data is a 1D array
    quantized_data = np.asarray(quantized_data).flatten()

    # Encode the quantized data into a digital format (e.g., binary)
    encoded_data = ''.join(format(int(sample), '08b') for sample in quantized_data)

    return encoded_data


# Función para graficar los primeros 100 bits de los datos codificados
def plot_encoded_data(encoded_data, num_bits=100):
    # Get the first num_bits of the encoded data
    bit_string = encoded_data[:num_bits]

    # Split the bit string into groups of 8 bits
    bytes_list = [bit_string[i:i+8] for i in range(0, len(bit_string), 8)]

    # Convert each byte to its decimal representation for plotting
    decimal_values = [int(byte, 2) for byte in bytes_list]

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(decimal_values)), decimal_values, color='blue')
    ax.set_xlabel('Byte Index')
    ax.set_ylabel('Decimal Value')
    ax.set_title('First 100 Bits of Encoded Data (Grouped by Byte)')
    plt.show()
    fig.savefig('./assets/results/encoded_data.png')


# Función para aplicar la Transformada de Fourier y graficar el espectro de frecuencias
def plot_fourier_transform(audio_data, samplerate):
    # Aplicar la Transformada de Fourier
    N = len(audio_data)
    yf = fft(audio_data)
    xf = fftfreq(N, 1 / samplerate)

    # Solo usar las frecuencias positivas
    xf = xf[:N // 2]
    yf = np.abs(yf[:N // 2])

    # Graficar el espectro de frecuencias
    fig, ax = plt.subplots()
    ax.plot(xf, yf)
    ax.set_title('Espectro de Frecuencias')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Amplitud')
    ax.grid()
    ax.set_xlim([min(xf), max(xf)])
    plt.show()
    fig.savefig('./assets/results/frequency_spectrum.png')

    # Retornar los valores transformados y las frecuencias correspondientes
    return xf, yf


# Función para graficar el histograma del espectro de frecuencias
def plot_frequency_histogram(frequencies, transformed_values, num_bins=50):
    # Calcular la magnitud de los valores transformados
    magnitudes = np.abs(transformed_values)

    # Filtrar las frecuencias positivas
    mask = frequencies >= 0
    positive_frequencies = frequencies[mask]
    positive_magnitudes = magnitudes[mask]

    # Crear el histograma: distribuye las frecuencias en diferentes intervalos de magnitud
    plt.figure(figsize=(10, 4))
    plt.hist(positive_frequencies, bins=num_bins, weights=positive_magnitudes, color='blue', edgecolor='black')
    plt.title('Histograma del Espectro de Frecuencias')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------


# ------------------------CODIGO PRINCIPAL------------------------------------------------------------------------------
# 1. Leer la nota de voz en formato WAV
data, samplerate = sf.read('./assets/test.wav')

# 2. Reproducir la nota de voz original
#sd.play(data, samplerate)
#status = sd.wait()

# 3. Graficar la nota de voz original
plot_signal(data, 'Nota de voz original', 'Tiempo', 'Amplitud', 'green')

# 3. Filtrar los intervalos de silecio
filtered_data = filter_silence(data, samplerate)

# 4. Reproducir la nota de voz filtrada
#sd.play(filtered_data, samplerate)
#status = sd.wait()

# 5. Graficar la nota de voz filtrada eliminando los intervalos de silencio
plot_signal(filtered_data, 'Nota de voz filtrada quitando silencio', 'Tiempo', 'Amplitud', 'blue')

# 6. Sampling, Quantization, and Coding
# 6.1. Number of quantization levels
num_levels = 256

new_samplerate = 16000  # New sample rate in Hz
resampled_data = resample_audio(filtered_data, samplerate, new_samplerate)

# 6.2. Quantize the resampled audio data
quantized_data = quantize(resampled_data, num_levels)

# 6.3. Encode the quantized data
encoded_data = encode(quantized_data)

# 7. Print the first 100 bits of the encoded data as an example
print(encoded_data[:100])
plot_encoded_data(encoded_data)

# 8. Fourier Transformation
frequencies, transformed_values = plot_fourier_transform(resampled_data, new_samplerate)

# 9. Visualizing the Frequency Histogram
plot_frequency_histogram(frequencies, transformed_values)

#Identifying the normal frequency and frequency range of the recorded voice