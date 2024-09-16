import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
import soundfile as sf
from pystoi import stoi

def audio_to_spectrogram(audio_path, output_image='spectrogram.png'):
    audio, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=256, n_mels=512, fmax=sr/2)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0)
    plt.close()
    return S, sr, S_dB

def process_spectrogram_image(input_image, output_image='modified_spectrogram.png'):
    # Leer la imagen en color (sin convertir a escala de grises)
    spectrogram_image = cv2.imread(input_image)

    # Aplicar filtro bilateral para reducir el ruido, manteniendo bordes
    denoised_spectrogram = cv2.bilateralFilter(spectrogram_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Convertir la imagen a espacio LAB para mejorar el canal de luminosidad
    lab = cv2.cvtColor(denoised_spectrogram, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Aplicar CLAHE en el canal L para mejorar el contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Combinar los canales LAB nuevamente y convertir a BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_contrast_spectrogram = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Guardar el espectrograma modificado manteniendo el color
    cv2.imwrite(output_image, enhanced_contrast_spectrogram)

    return enhanced_contrast_spectrogram


def image_to_spectrogram(original_spectrogram, mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.array(mask, dtype=np.float32) / 255
    # Reshape the mask to match the spectrogram dimensions
    mask = cv2.resize(mask, dsize=(original_spectrogram.shape[1], original_spectrogram.shape[0]))
    masked_spectrogram = original_spectrogram * mask
    return masked_spectrogram

def spectrogram_to_audio(S, sr, output_audio='reconstructed_audio.wav'):
    reconstructed_audio = librosa.feature.inverse.mel_to_audio(S, sr=sr, n_iter=500, hop_length=256)
    sf.write(output_audio, reconstructed_audio, sr)

def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def main(audio_path):
    S, sr, S_dB = audio_to_spectrogram(audio_path)
    process_spectrogram_image('spectrogram.png')
    modified_S = image_to_spectrogram(S, 'modified_spectrogram.png')

    # Guardar el audio original
    spectrogram_to_audio(S, sr, 'reconstructed_audio_original.wav')

    # Guardar el audio modificado
    spectrogram_to_audio(modified_S, sr, 'reconstructed_audio_modified.wav')

    # Evaluar la diferencia entre los dos audios
    original_audio, _ = librosa.load('reconstructed_audio_original.wav', sr=sr)
    modified_audio, _ = librosa.load('reconstructed_audio_modified.wav', sr=sr)
    difference = np.sum((original_audio - modified_audio) ** 2)
    print(f'Difference in energy between original and modified audio: {difference}')

    # Calcular SNR para ambos audios
    noise_original = original_audio - modified_audio
    snr_original = calculate_snr(original_audio, noise_original)
    snr_modified = calculate_snr(modified_audio, noise_original)

    print(f'SNR of original audio: {snr_original} dB')
    print(f'SNR of modified audio: {snr_modified} dB')

    if snr_original > snr_modified:
        print('The original audio has better quality.')
    else:
        print('The modified audio has better quality.')

    stoi_score = stoi(original_audio, modified_audio, sr, extended=False)
    print(f'STOI Score: {stoi_score}')


if __name__ == "__main__":
    audio_path = '../AnalogToDigitalConversionVoice/assets/records/test4.wav'
    main(audio_path)