import os
from abc import ABC

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .base_processor import BaseProcessor
from .utils import ensure_directory


class ImageProcessor(BaseProcessor, ABC):

    def __init__(self, data_dir, report_dir):
        self.data_dir = data_dir
        self.report_dir = report_dir
        ensure_directory(self.data_dir)
        ensure_directory(self.report_dir)
        self.images = []
        self.labels = []
        self.features = None
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load_data(self):
        print(f"Loading images from {self.data_dir}...")
        # Obtener la lista de subcarpetas (nombres de animales)
        animal_folders = [folder for folder in os.listdir(self.data_dir)
                          if os.path.isdir(os.path.join(self.data_dir, folder))]

        for animal in animal_folders:
            animal_path = os.path.join(self.data_dir, animal)
            image_files = [f for f in os.listdir(animal_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            for file in image_files:
                img_path = os.path.join(animal_path, file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    self.images.append(img)
                    self.labels.append(animal)  # Usamos el nombre de la subcarpeta como etiqueta
                    print(f"Imagen {img_path} cargada exitosamente.")
                    print(f"Etiqueta: {self.labels[-1]}")
                except Exception as e:
                    print(f"Error al cargar la imagen {img_path}: {e}")
        print(f"Se cargaron {len(self.images)} imágenes en total.")
        print(self.labels)

    def preprocess_data(self):
        # No es necesario preprocesar las imágenes en este caso
        pass

    def extract_features(self):
        print("Extracting features from images...")
        features = []
        for img in self.images:
            img_resized = img.resize((64, 64))
            img_array = np.array(img_resized)
            img_flat = img_array.flatten()
            features.append(img_flat)
        self.features = np.array(features)

    def train_model(self):
        print("Training the model...")
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.model.fit(X_train, y_train)

    def evaluate_model(self):
        print("Evaluating the model...")
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print("=== Classification Report ===")
        print(report)
        with open(os.path.join(self.report_dir, 'image_classification_report.txt'), 'w') as f:
            f.write(report)

    def visualize_data(self):
        print("Visualizing sample images...")
        plt.figure(figsize=(10, 5))
        for i in range(min(5, len(self.images))):
            plt.subplot(1, 5, i + 1)
            plt.imshow(self.images[i])
            plt.title(self.labels[i])
            plt.axis('off')
        plt.suptitle('Sample Images')
        plt.show()

    def generate_report(self):
        print("Generating report...")
        ensure_directory(self.report_dir)
        self.visualize_data()
        self.extract_features()
        self.train_model()
        self.evaluate_model()
        print("=== Report generated successfully ===")