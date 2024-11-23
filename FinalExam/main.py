from media_classifier.image_processor import ImageProcessor
from media_classifier.audio_processor import AudioProcessor
from media_classifier.text_processor import TextProcessor
from media_classifier.utils import ensure_directory


def load_custom_stopwords(file_path):
    """
    Carga una lista de stopwords personalizadas desde un archivo de texto.

    :param file_path: Ruta al archivo de stopwords personalizadas.
    :return: Lista de palabras.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]


def main():
    data_dir = 'data'
    report_dir = 'reports'
    ensure_directory(report_dir)

    # Image Processor
    print("=== Image Processor ===")
    image_processor = ImageProcessor(
        data_dir=f'{data_dir}/images/coil-20',
        report_dir=report_dir
    )
    image_processor.generate_report()

    # Audio Processor
    print("\n=== Audio Processor ===")
    audio_processor = AudioProcessor(
        data_dir=f'{data_dir}/audio/ESC-50',
        report_dir=report_dir,
        noise_factor=0.005
    )
    audio_processor.load_data()
    if audio_processor.audios:
        audio_processor.generate_report()
        audio_processor.visualize_data()
    else:
        print("No audio files found in the data directory.")

    # Cargar stopwords personalizadas
    custom_stopwords = load_custom_stopwords('./data/stop_words/stop_words')

    # Text Processor
    print("\n=== Text Processor ===")
    text_processor = TextProcessor(
        data_dir=f'{data_dir}/text/20-Newsgroups',
        report_dir=report_dir,
        custom_stopwords=custom_stopwords
    )
    text_processor.generate_report()


if __name__ == '__main__':
    main()