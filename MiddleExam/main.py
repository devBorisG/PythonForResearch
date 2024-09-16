import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
import soundfile as sf

def audio_to_spectrogram(audio_path, output_image='spectrogram.png'):
    audio, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=256, n_mels=256)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0)
    plt.close()
    return S, sr, S_dB

def process_spectrogram_image(input_image, output_image='modified_spectrogram.png'):
    spectrogram_image = cv2.imread(input_image)
    blurred_spectrogram = cv2.GaussianBlur(spectrogram_image, (5, 5), 0)
    cv2.imwrite('blurred_spectrogram.png', blurred_spectrogram)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_spectrogram = cv2.filter2D(blurred_spectrogram, -1, kernel)
    cv2.imwrite('sharpened_spectrogram.png', sharpened_spectrogram)
    edges = cv2.Canny(sharpened_spectrogram, 100, 200)
    cv2.imwrite('edges_spectrogram.png', edges)
    cv2.imwrite(output_image, edges)
    return edges

def image_to_spectrogram(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    S = np.array(img, dtype=np.float32)
    S = librosa.db_to_power(S)
    return S

def spectrogram_to_audio(S, sr, output_audio='reconstructed_audio.wav'):
    mel_spec = librosa.feature.inverse.mel_to_stft(S, sr=sr)
    reconstructed_audio = librosa.griffinlim(mel_spec, n_iter=200, hop_length=512)
    sf.write(output_audio, reconstructed_audio, sr)

def main(audio_path):
    S, sr, S_dB = audio_to_spectrogram(audio_path)
    process_spectrogram_image('spectrogram.png')
    modified_S = image_to_spectrogram('modified_spectrogram.png')
    spectrogram_to_audio(modified_S, sr, 'reconstructed_audio_modified.wav')

if __name__ == "__main__":
    audio_path = '../AnalogToDigitalConversionVoice/assets/records/test4.wav'
    main(audio_path)