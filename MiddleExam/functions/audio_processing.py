import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def audio_to_spectrogram(audio_file, output_image='spectrogram.png'):
    # Cargar archivo de audio
    audio, sr = librosa.load(audio_file)

    # Convertir a un espectrograma mel
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Guardar el espectrograma como imagen
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()

    return S, sr


def spectrogram_to_audio(S, sr, output_audio='reconstructed_audio.wav'):
    """
    Reconstructs audio from a spectrogram and saves it to a file.

    Args:
        S (numpy.ndarray): Spectrogram.
        sr (int): Sample rate.
        output_file (str): Path to save the reconstructed audio file.

    Returns:
        None
    """
    # Convert the spectrogram to a mel spectrogram
    mel_spec = librosa.feature.inverse.mel_to_stft(S, sr=sr)

    # Reconstruct the audio signal from the mel spectrogram
    reconstructed_audio = librosa.griffinlim(mel_spec)

    # Save the reconstructed audio to a file
    sf.write(output_audio, reconstructed_audio, sr)


def dummy():
    print("dummy audio processing function")
