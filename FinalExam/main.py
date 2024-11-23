from media_classifier.image_processor import ImageProcessor
from media_classifier.utils import ensure_directory


def main():
    data_dir = 'data'
    report_dir = 'reports'
    ensure_directory(report_dir)

    # Image Processor
    print("=== Image Processor ===")
    image_processor = ImageProcessor(f'{data_dir}/images/animals', report_dir)
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


if __name__ == '__main__':
    main()