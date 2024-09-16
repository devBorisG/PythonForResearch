# MiddleExam/functions/audio_processing.py
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import cv2

def audio_to_spectrogram(audio_file, output_image='spectrogram.png'):
    # Cargar archivo de audio
    audio, sr = librosa.load(audio_file)

    # Convertir a un espectrograma mel
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Guardar el espectrograma como imagen
    plt.figure(figsize=(10, 4))
    plt.axis('off')  # Ocultar los ejes
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0)
    plt.close()

    return S, sr

def image_to_spectrogram(image_path):
    # Leer la imagen del espectrograma
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convertir la imagen a un espectrograma
    S = np.array(img, dtype=np.float32)
    S = librosa.db_to_power(S)

    return S

def spectrogram_to_audio(S, sr, output_audio='reconstructed_audio.wav'):
    # Convert the spectrogram to a mel spectrogram
    mel_spec = librosa.feature.inverse.mel_to_stft(S, sr=sr)

    # Reconstruct the audio signal from the mel spectrogram
    reconstructed_audio = librosa.griffinlim(mel_spec, n_iter=64)  # Aumentar el número de iteraciones

    # Save the reconstructed audio to a file
    sf.write(output_audio, reconstructed_audio, sr)