from media_classifier.image_processor import ImageProcessor
from media_classifier.audio_processor import AudioProcessor
from media_classifier.text_processor import TextProcessor
from media_classifier.utils import ensure_directory


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
    image_processor.load_data()
    if image_processor.images:
        image_processor.preprocess_data()
        image_processor.extract_features()
        image_processor.train_model()
        image_processor.evaluate_model()
        image_processor.visualize_data()
        image_processor.generate_report()
    else:
        print("No images found in the data directory.")

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


if __name__ == '__main__':
    main()