import functions.image_processing as ip
import functions.audio_processing as ap

# TODO: 1. Audio to Spectrogram Conversion:
# 1.1.Convert the input audio file (e.g., WAV or MP3 format) into a spec-
# trogram, which visually represents the frequency and amplitude of
# the sound over time.
# 1.2. Save the spectrogram as an image (e.g., PNG or JPG format).

# TODO: 2. Image Processing on the Spectrogram:
# 2.1. Load the spectrogram image and apply image filtering or enhance-
# ment techniques such as blurring, sharpening, edge detection, or noise, etc.
# 2.2. Perform image segmentation or thresholding to alter specific parts of
# the spectrogram.

# TODO: 3. Reconstruction of the Audio:
# 3.1. After modifying the spectrogram image, reconstruct the audio file
# from the modified image
# 3.2. Save the reconstructed audio file (e.g., WAV or MP3 format).

# TODO: 4. Comparison:
# 4.1. Compare the original and reconstructed audio to assess the impact
# of the image processing on the sound quality.


def main():
    # 1. Convertir audio a espectrograma
    S, sr = ap.audio_to_spectrogram('../AnalogToDigitalConversionVoice/assets/records/test4.wav', 'spectrogram.png')

    # 2. Procesar la imagen del espectrograma
    ip.process_spectrogram_image('spectrogram.png', 'modified_spectrogram.png')

    # 3. Reconstruir audio desde el espectrograma modificado
    ap.spectrogram_to_audio(S, sr, 'reconstructed_audio.wav')


if __name__ == "__main__":
    main()
