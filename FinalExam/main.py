from media_classifier.image_processor import ImageProcessor
from media_classifier.audio_processor import AudioProcessor
from media_classifier.text_processor import TextProcessor
from media_classifier.utils import ensure_directory


def load_custom_stopwords(file_path):
    """
    Load a list of custom stopwords from a text file.

    Args:
        file_path (str): Path to the custom stopwords file.

    Returns:
        list: List of stopwords as strings.
    """
    # Open the stopwords file in read mode with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read each line, strip whitespace, and convert to lowercase
        return [line.strip().lower() for line in f if line.strip()]


def main():
    """
    Main function that orchestrates the processing of images, audio, and text.

    This function initializes the respective processors, loads data, and generates
    reports and visualizations for each media type.
    """
    # Define data and report directories
    data_dir = 'data'       # Base directory containing the data
    report_dir = 'reports'  # Directory where reports and outputs will be saved

    # Ensure the report directory exists, creating it if necessary
    ensure_directory(report_dir)

    # ==============================
    # Image Processing
    # ==============================
    print("=== Image Processor ===")
    image_processor = ImageProcessor(
        data_dir=f'{data_dir}/images/coil-20',  # Path to the image dataset
        report_dir=report_dir                   # Directory to save image reports
    )
    # Generate the image processing report
    image_processor.generate_report()

    # ==============================
    # Audio Processing
    # ==============================
    print("\n=== Audio Processor ===")
    audio_processor = AudioProcessor(
        data_dir=f'{data_dir}/audio/ESC-50',  # Path to the audio dataset
        report_dir=report_dir,                # Directory to save audio reports
        noise_factor=0.005                    # Factor to inject noise into audio signals
    )
    # Load audio data from the specified directory
    audio_processor.load_data()
    # Check if any audio files were loaded successfully
    if audio_processor.audios:
        # Generate the audio processing report
        audio_processor.generate_report()
        # Visualize audio data (waveforms and spectrograms)
        audio_processor.visualize_data()
    else:
        print("No audio files found in the data directory.")

    # ==============================
    # Text Processing
    # ==============================

    # Load custom stopwords from a text file
    custom_stopwords = load_custom_stopwords('./data/stop_words/stop_words')

    print("\n=== Text Processor ===")
    text_processor = TextProcessor(
        data_dir=f'{data_dir}/text/20-Newsgroups',  # Path to the text dataset
        report_dir=report_dir,                      # Directory to save text reports
        custom_stopwords=custom_stopwords           # List of custom stopwords
    )
    # Generate the text processing report
    text_processor.generate_report()


if __name__ == '__main__':
    main()
