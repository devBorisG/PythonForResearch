# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import resample
import sounddevice as sd
from numpy.fft import fft, fftfreq
import parselmouth


# ------------------------DEFINICION DE FUNCIONES-----------------------------------------------------------------------
# Funcion para realizar graficos con matplotlib
def plot_signal(data, title, xlabel, ylabel, color):
    """
    Grafica una señal de audio utilizando matplotlib.

    Args:
        data (numpy.ndarray): Datos de la señal de audio.
        title (str): Título del gráfico.
        xlabel (str): Etiqueta del eje X.
        ylabel (str): Etiqueta del eje Y.
        color (str): Color de la línea del gráfico.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    ax.plot(data, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    fig.savefig('./assets/results/' + title + '.png')
    plt.close(fig)


# Función para detectar y filtrar intervalos de silencio
def filter_silence(audio_data, samplerate, silence_threshold=0.01, min_silence_duration=0.5):
    """
    Grafica una señal de audio utilizando matplotlib.

    Args:
        data (numpy.ndarray): Datos de la señal de audio.
        title (str): Título del gráfico.
        xlabel (str): Etiqueta del eje X.
        ylabel (str): Etiqueta del eje Y.
        color (str): Color de la línea del gráfico.

    Returns:
        None
    """
    # Convertir la duración mínima de silencio a muestras
    min_silence_samples = int(min_silence_duration * samplerate)

    # Si el audio tiene múltiples canales, procesar cada canal por separado
    if len(audio_data.shape) > 1:
        filtered_channels = []
        for channel in range(audio_data.shape[1]):
            filtered_channel = filter_silence(audio_data[:, channel], samplerate, silence_threshold,
                                              min_silence_duration)
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
    """
    Realiza el remuestreo de una señal de audio a una nueva frecuencia de muestreo.

    Args:
        audio_data (numpy.ndarray): Datos de la señal de audio.
        original_samplerate (int): Frecuencia de muestreo original.
        new_samplerate (int): Nueva frecuencia de muestreo.

    Returns:
        numpy.ndarray: Datos de la señal de audio remuestreados.
    """
    # Calculate the number of samples in the resampled audio
    num_samples = int(len(audio_data) * new_samplerate / original_samplerate)

    # Resample the audio data
    resampled_data = resample(audio_data, num_samples)

    return resampled_data


# Función para realizar la cuantificación
def quantize(audio_data, num_levels):
    """
    Realiza la cuantificación de una señal de audio.

    Args:
        audio_data (numpy.ndarray): Datos de la señal de audio.
        num_levels (int): Número de niveles de cuantificación.

    Returns:
        numpy.ndarray: Datos de la señal de audio cuantificados.
    """
    # Normalize the audio data to the range [-1, 1]
    audio_data_normalized = 2 * (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data)) - 1

    # Quantize the normalized data
    quantized_data = np.round((audio_data_normalized + 1) * (num_levels - 1) / 2)

    return quantized_data


# Función para realizar la codificación
def encode(quantized_data):
    """
    Codifica los datos cuantificados en un formato digital (binario).

    Args:
        quantized_data (numpy.ndarray): Datos de la señal de audio cuantificados.

    Returns:
        str: Datos codificados en formato binario.
    """
    # Ensure quantized_data is a 1D array
    quantized_data = np.asarray(quantized_data).flatten()

    # Encode the quantized data into a digital format (e.g., binary)
    encoded_data = ''.join(format(int(sample), '08b') for sample in quantized_data)

    return encoded_data


# Función para graficar los primeros 100 bits de los datos codificados
def plot_encoded_data(encoded_data, num_bits=100):
    """
    Grafica los primeros bits de los datos codificados.

    Args:
        encoded_data (str): Datos codificados en formato binario.
        num_bits (int): Número de bits a graficar.

    Returns:
        None
    """
    # Get the first num_bits of the encoded data
    bit_string = encoded_data[:num_bits]

    # Split the bit string into groups of 8 bits
    bytes_list = [bit_string[i:i + 8] for i in range(0, len(bit_string), 8)]

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
    plt.close(fig)


# Función para aplicar la Transformada de Fourier y graficar el espectro de frecuencias
def plot_fourier_transform(audio_data, samplerate):
    """
    Aplica la Transformada de Fourier y grafica el espectro de frecuencias.

    Args:
        audio_data (numpy.ndarray): Datos de la señal de audio.
        samplerate (int): Frecuencia de muestreo de la señal de audio.

    Returns:
        tuple: Frecuencias y valores transformados.
    """
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
    plt.close(fig)

    # Retornar los valores transformados y las frecuencias correspondientes
    return xf, yf


# Función para graficar el histograma del espectro de frecuencias
def plot_frequency_histogram(frequencies, transformed_values, num_bins=150):
    """
    Grafica el histograma del espectro de frecuencias.

    Args:
        frequencies (numpy.ndarray): Frecuencias obtenidas de la Transformada de Fourier.
        transformed_values (numpy.ndarray): Valores transformados de la señal de audio.
        num_bins (int): Número de bins para el histograma.

    Returns:
        None
    """
    # Calcular la magnitud de los valores transformados
    magnitudes = np.abs(transformed_values)

    # Filtrar las frecuencias positivas
    mask = frequencies >= 0
    positive_frequencies = frequencies[mask]
    positive_magnitudes = magnitudes[mask]

    # Crear el histograma: distribuye las frecuencias en diferentes intervalos de magnitud
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(positive_frequencies, bins=num_bins, weights=positive_magnitudes, color='blue', edgecolor='black')
    ax.set_title('Histograma del Espectro de Frecuencias')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Amplitud')
    ax.grid(True)
    plt.show()
    fig.savefig('./assets/results/frequency_histogram.png')
    plt.close(fig)


# Funcion para identificar la frecuencia normal y el rango de frecuencia de la voz grabada
def identify_normal_frequency(frequencies, transformed_values):
    """
    Identifica la frecuencia normal y el rango de frecuencia de la voz grabada.

    Args:
        frequencies (numpy.ndarray): Frecuencias obtenidas de la Transformada de Fourier.
        transformed_values (numpy.ndarray): Valores transformados de la señal de audio.

    Returns:
        tuple: Frecuencia normal, frecuencia mínima y frecuencia máxima.
    """
    # Calcular la magnitud de los valores transformados
    magnitudes = np.abs(transformed_values)

    # Filtrar las frecuencias positivas
    mask = frequencies >= 0
    positive_frequencies = frequencies[mask]
    positive_magnitudes = magnitudes[mask]

    # Calcular la frecuencia normal
    normal_frequency = positive_frequencies[np.argmax(positive_magnitudes)]
    print('Frecuencia normal:', normal_frequency, 'Hz')

    # Calcular el rango de frecuencias
    min_frequency = positive_frequencies[np.argmin(positive_magnitudes)]
    max_frequency = positive_frequencies[np.argmax(positive_magnitudes)]
    print('Rango de frecuencias:', min_frequency, 'Hz -', max_frequency, 'Hz')

    return normal_frequency, min_frequency, max_frequency


# Funcion para graficar las frecuencias dominantes
def plot_dominant_frequencies(audio_data, samplerate, num_frequencies=5):
    """
    Grafica las frecuencias más predominantes en una señal de audio.

    Args:
        audio_data (numpy.ndarray): Datos de la señal de audio.
        samplerate (int): Frecuencia de muestreo de la señal de audio.
        num_frequencies (int): Número de frecuencias predominantes a graficar.

    Returns:
        None
    """
    # Aplicar la Transformada de Fourier
    N = len(audio_data)
    yf = fft(audio_data)
    xf = fftfreq(N, 1 / samplerate)

    # Solo usar las frecuencias positivas
    xf = xf[:N // 2]
    yf = np.abs(yf[:N // 2])

    # Identificar las frecuencias más predominantes
    indices = np.argsort(yf)[-num_frequencies:]
    dominant_frequencies = xf[indices]
    dominant_amplitudes = yf[indices]

    # Graficar las frecuencias predominantes
    fig, ax = plt.subplots()
    ax.bar(dominant_frequencies, dominant_amplitudes, color='red')
    ax.set_title('Frecuencias Más Predominantes')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Amplitud')
    plt.show()
    fig.savefig('./assets/results/dominant_frequencies.png')
    plt.close(fig)


# Función para detectar formantes
def detect_formants(audio_data, samplerate):
    """
    Detecta los formantes en una señal de audio.

    Args:
        audio_data (numpy.ndarray): Datos de la señal de audio.
        samplerate (int): Frecuencia de muestreo de la señal de audio.

    Returns:
        numpy.ndarray: Formantes detectados.
    """
    pre_emphasis = 0.97
    emphasized_signal = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = frame_size * samplerate, frame_stride * samplerate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (samplerate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / samplerate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks


# Funcion para detectar vocales
def detect_vowels(audio_data, samplerate):
    """
    Detecta las vocales en una señal de audio.

    Args:
        audio_data (numpy.ndarray): Datos de la señal de audio.
        samplerate (int): Frecuencia de muestreo de la señal de audio.

    Returns:
        numpy.ndarray: Vocales detectadas.
    """
    # Convertir el audio a un objeto Sound de parselmouth
    sound = parselmouth.Sound(audio_data, samplerate)

    # Obtener los formantes utilizando parselmouth
    formant = sound.to_formant_burg()

    # Inicializar una lista para almacenar las vocales detectadas
    vowels = []

    # Iterar sobre cada frame para obtener los formantes
    for t in np.arange(0, sound.duration, 0.01):
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)

        # Filtrar los formantes para identificar vocales
        if f1 and f2:
            if 300 < f1 < 900 and 850 < f2 < 2500:
                vowels.append((f1, f2))

    return np.array(vowels)


# Función para identificar la vocal basada en los formantes
vowel_formants_range = {
    'a': {'F1': (600, 900), 'F2': (850, 1500)},
    'e': {'F1': (400, 700), 'F2': (1650, 2500)},
    'i': {'F1': (200, 450), 'F2': (1700, 2500)},
    'o': {'F1': (400, 700), 'F2': (700, 1200)},
    'u': {'F1': (250, 450), 'F2': (600, 1100)}
}


def identify_vowel(f1, f2, vowel_formants=vowel_formants_range):
    """
    Identifica la vocal basada en los formantes.

    Args:
        f1 (float): Primer formante.
        f2 (float): Segundo formante.
        vowel_formants (dict): Rango de formantes para cada vocal.

    Returns:
        str: Vocal identificada.
    """
    for vowel, formants in vowel_formants.items():
        if formants['F1'][0] <= f1 <= formants['F1'][1] and formants['F2'][0] <= f2 <= formants['F2'][1]:
            return vowel
    return 'Unknown'


# Función para calcular la energía
def calculate_energy(audio_data, frame_size, frame_stride, samplerate):
    """
    Calcula la energía de una señal de audio.

    Args:
        audio_data (numpy.ndarray): Datos de la señal de audio.
        frame_size (float): Tamaño del frame en segundos.
        frame_stride (float): Desplazamiento del frame en segundos.
        samplerate (int): Frecuencia de muestreo de la señal de audio.

    Returns:
        numpy.ndarray: Energía de la señal de audio.
    """
    frame_length, frame_step = frame_size * samplerate, frame_stride * samplerate
    signal_length = len(audio_data)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(audio_data, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    energy = np.sum(frames ** 2, axis=1)
    return energy


# Función para analizar el pitch
def analyze_pitch(audio_data, samplerate):
    """
    Analiza el pitch de una señal de audio.

    Args:
        audio_data (numpy.ndarray): Datos de la señal de audio.
        samplerate (int): Frecuencia de muestreo de la señal de audio.

    Returns:
        numpy.ndarray: Valores de pitch de la señal de audio.
    """
    sound = parselmouth.Sound(audio_data, samplerate)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan  # Replace unvoiced parts with NaN
    return pitch_values


def plot_formants(formants, title='Formantes'):
    """
    Grafica los formantes de una señal de audio.

    Args:
        formants (numpy.ndarray): Formantes detectados.
        title (str): Título del gráfico.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    for i, formant in enumerate(formants.T[:2]):
        plt.plot(formant, label=f'Formante {i+1}')
    plt.title(title)
    plt.xlabel('Tiempo (frames)')
    plt.ylabel('Frecuencia (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('./assets/results/formants.png')
    plt.close()


def plot_energy(energy, title='Energía de la Señal'):
    """
    Grafica la energía de una señal de audio.

    Args:
        energy (numpy.ndarray): Energía de la señal de audio.
        title (str): Título del gráfico.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(energy, color='blue')
    plt.title(title)
    plt.xlabel('Tiempo (frames)')
    plt.ylabel('Energía')
    plt.grid(True)
    plt.show()
    plt.savefig('./assets/results/energy.png')
    plt.close()


def plot_pitch(pitch_values, title='Pitch de la Señal'):
    """
    Grafica el pitch de una señal de audio.

    Args:
        pitch_values (numpy.ndarray): Valores de pitch de la señal de audio.
        title (str): Título del gráfico.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pitch_values, color='red')
    plt.title(title)
    plt.xlabel('Tiempo (frames)')
    plt.ylabel('Frecuencia (Hz)')
    plt.grid(True)
    plt.show()
    plt.savefig('./assets/results/pitch.png')
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------


# ------------------------CODIGO PRINCIPAL------------------------------------------------------------------------------
# 1. Leer la nota de voz en formato WAV
data, samplerate = sf.read('assets/records/test4.wav')

# Si el audio tiene múltiples canales, promediar los canales
if len(data.shape) > 1:
    data = np.mean(data, axis=1)

# Ahora `data` es un array unidimensional

# 2. Reproducir la nota de voz original
#sd.play(data, samplerate)
#status = sd.wait()

# 3. Graficar la nota de voz original
plot_signal(data, 'Nota de voz original', 'Tiempo', 'Amplitud', 'green')

# 4. Filtrar los intervalos de silencio
filtered_data = filter_silence(data, samplerate)

# 5. Reproducir la nota de voz filtrada
#sd.play(filtered_data, samplerate)
#status = sd.wait()

# 6. Graficar la nota de voz filtrada eliminando los intervalos de silencio
plot_signal(filtered_data, 'Nota de voz filtrada quitando silencio', 'Tiempo', 'Amplitud', 'blue')

# 7. Sampling, Quantization, and Coding
# 7.1. Number of quantization levels
num_levels = 256

new_samplerate = 4000  # New sample rate in Hz (e.g., 4000 Hz), to resample the audio data to a lower rate for
# Nyquist theorem
resampled_data = resample_audio(filtered_data, samplerate, new_samplerate)

# 7.2. Quantize the resampled audio data
quantized_data = quantize(resampled_data, num_levels)

# 7.3. Encode the quantized data
encoded_data = encode(quantized_data)

# 8. Print the first 100 bits of the encoded data as an example
print(encoded_data[:100])
plot_encoded_data(encoded_data)

# 9. Fourier Transformation
frequencies, transformed_values = plot_fourier_transform(resampled_data, new_samplerate)

# 10. Visualizing the Frequency Histogram
plot_frequency_histogram(frequencies, transformed_values)

# 11. Identifying the normal frequency and frequency range of the recorded voice
identify_normal_frequency(frequencies, transformed_values)

# 12. Plotting the Dominant Frequencies
plot_dominant_frequencies(resampled_data, new_samplerate)

# 13. Detectar formantes para el analisis de vocales
formants = detect_formants(filtered_data, samplerate)
plot_formants(formants)

# 14. Vocales analizadas en la nota de voz
vowels = detect_vowels(filtered_data, samplerate)
identify_vowels = [identify_vowel(f1, f2) for f1, f2 in vowels]
print("Vocales identificadas:", identify_vowels)

# 15. Calcular la energía de la señal
frame_size = 0.025
frame_stride = 0.01
energy = calculate_energy(filtered_data, frame_size, frame_stride, samplerate)
plot_energy(energy)

# 16. Analizar el pitch
pitch_values = analyze_pitch(filtered_data, samplerate)
plot_pitch(pitch_values)
