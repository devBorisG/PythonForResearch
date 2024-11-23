import os
from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from skimage import feature
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
        self.features = []
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.le = LabelEncoder()  # Codificador de etiquetas

    def load_data(self):
        print(f"Cargando imágenes desde {self.data_dir}...")
        image_files = [f for f in os.listdir(self.data_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        for file in image_files:
            img_path = os.path.join(self.data_dir, file)
            try:
                # Usar OpenCV para leer la imagen
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
                    self.images.append(img)
                    # Extraer la etiqueta del nombre del archivo
                    label = str(file.split('__')[0]).strip().lower()  # Asumiendo que el formato es 'objX__Y.png'
                    self.labels.append(label)
                    print(f"Imagen {img_path} cargada exitosamente.")
                    print(f"Etiqueta: {label}")
                else:
                    print(f"Error al cargar la imagen {img_path}: La imagen es None")
            except Exception as e:
                print(f"Error al cargar la imagen {img_path}: {e}")
        print(f"Se cargaron {len(self.images)} imágenes en total.")
        print(f"Etiquetas: {self.labels}")

    def preprocess_data(self):
        # No es necesario preprocesar las imágenes en este caso
        pass

    def extract_features(self):
        print("Extrayendo características de las imágenes...")
        features = []
        for img in self.images:
            # Redimensionar la imagen si es necesario
            # img = cv2.resize(img, (128, 128))
            # Extraer características de color y textura
            color_features = self.extract_color_histogram(img)
            texture_features = self.extract_lbp_features(img)
            combined_features = np.hstack([color_features, texture_features])
            features.append(combined_features)
        self.features = np.array(features)
        # Codificar etiquetas
        self.labels = self.le.fit_transform(self.labels)
        print(f"Características extraídas: {self.features.shape}")

    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        # Convertir la imagen a espacio de color HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Calcular el histograma
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                            [0, 180, 0, 256, 0, 256])
        # Normalizar el histograma
        cv2.normalize(hist, hist)
        # Aplanar el histograma
        return hist.flatten()

    def extract_lbp_features(self, image, numPoints=24, radius=8):
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calcular LBP
        lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
        # Calcular el histograma
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))
        # Normalizar el histograma
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def train_model(self):
        print("Entrenando el modelo...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42, stratify=self.labels)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.model.fit(X_train, y_train)
        print("Modelo entrenado exitosamente.")

    def evaluate_model(self):
        print("Evaluando el modelo...")
        y_pred = self.model.predict(self.X_test)
        target_names = [str(name) for name in self.le.classes_]
        print(f"tipos de target_names: {type(target_names)}, elementos: {target_names}")
        report = classification_report(self.y_test, y_pred, target_names=target_names)
        print("=== Reporte de Clasificación ===")
        print(report)
        with open(os.path.join(self.report_dir, 'image_classification_report.txt'), 'w') as f:
            f.write(report)

    def visualize_data(self):
        print("Visualizando imágenes de muestra...")
        plt.figure(figsize=(10, 5))
        for i in range(min(5, len(self.images))):
            plt.subplot(1, 5, i + 1)
            plt.imshow(self.images[i])
            plt.title(self.le.inverse_transform([self.labels[i]])[0])
            plt.axis('off')
        plt.suptitle('Imágenes de muestra')
        plt.show()

    def generate_report(self):
        print("Generando reporte...")
        ensure_directory(self.report_dir)
        self.extract_features()
        self.train_model()
        self.evaluate_model()
        print("=== Reporte generado exitosamente ===")
