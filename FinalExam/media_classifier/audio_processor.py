# media_classifier/audio_processor.py

import os
from abc import ABC

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
from .base_processor import BaseProcessor
from .utils import ensure_directory
import seaborn as sns


class AudioProcessor(BaseProcessor, ABC):
    def __init__(self, data_dir, report_dir, noise_factor=0.005):
        """
        Inicializa el procesador de audio.

        :param data_dir: Directorio que contiene los archivos de audio.
        :param report_dir: Directorio donde se guardarán los reportes y resultados.
        :param noise_factor: Factor para la inyección de ruido.
        """
        self.data_dir = data_dir
        self.report_dir = report_dir
        self.noise_factor = noise_factor
        ensure_directory(self.report_dir)
        self.images_dir = os.path.join(self.report_dir, 'images', 'audio_processor')
        self.audios = []
        self.file_names = []
        self.pitch_levels = []
        self.noise_levels = []
        self.durations = []  # Lista para almacenar duraciones
        self.results_df = pd.DataFrame()

    def load_data(self):
        print(f"Cargando audios desde {self.data_dir}...")
        audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
        audio_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(audio_extensions)]

        if not audio_files:
            print("No se encontraron archivos de audio en el directorio especificado.")
            return

        for file in audio_files:
            file_path = os.path.join(self.data_dir, file)
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                self.durations.append(duration)
                # Inyectar ruido si se especifica
                if self.noise_factor > 0:
                    y = self.add_noise(y)
                self.audios.append((y, sr))
                self.file_names.append(file)
                print(f"Audio {file_path} cargado exitosamente. Duración: {duration:.2f} segundos.")
            except Exception as e:
                print(f"Error al cargar el audio {file_path}: {e}")
        print(f"Se cargaron {len(self.audios)} audios en total.")

    def add_noise(self, y):
        """
        Agrega ruido blanco gaussiano a una señal de audio.

        :param y: Señal de audio original.
        :return: Señal de audio con ruido.
        """
        noise = np.random.randn(len(y))
        y_noisy = y + self.noise_factor * noise
        return y_noisy

    def estimate_pitch(self, y, sr):
        """
        Estima el pitch dominante de una señal de audio usando librosa.

        :param y: Señal de audio.
        :param sr: Tasa de muestreo.
        :return: Pitch dominante en Hz.
        """
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = []
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch_val = pitches[index, i]
                if pitch_val > 0:
                    pitch.append(pitch_val)
            if pitch:
                dominant_pitch = np.median(pitch)
                return dominant_pitch
            else:
                return 0
        except Exception as e:
            print(f"Error al estimar pitch: {e}")
            return 0

    def calculate_noise_level(self, y):
        """
        Calcula el nivel de ruido usando RMS (Root Mean Square).

        :param y: Señal de audio.
        :return: Nivel de ruido (RMS).
        """
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        return rms_mean

    def extract_features(self):
        print("Extrayendo características de los audios...")
        for idx, (y, sr) in enumerate(self.audios):
            pitch = self.estimate_pitch(y, sr)
            noise_level = self.calculate_noise_level(y)
            self.pitch_levels.append(pitch)
            self.noise_levels.append(noise_level)
            print(f"Audio {self.file_names[idx]}: Pitch = {pitch:.2f} Hz, Noise Level (RMS) = {noise_level:.5f}")

        # Crear DataFrame con los resultados
        self.results_df = pd.DataFrame({
            'file_name': self.file_names,
            'dominant_pitch_Hz': self.pitch_levels,
            'noise_level_RMS': self.noise_levels,
            'duration_seconds': self.durations
        })
        print("Características extraídas y almacenadas en DataFrame.")

    def save_results(self):
        """
        Guarda los resultados en un archivo CSV.
        """
        csv_path = os.path.join(self.report_dir, 'audio_features.csv')
        self.results_df.to_csv(csv_path, index=False)
        print(f"Resultados guardados en {csv_path}")

    def statistical_summary(self):
        """
        Genera resúmenes estadísticos básicos de las características de audio.
        """
        print("Generando resúmenes estadísticos de las características de audio...")
        summary = self.results_df.describe()
        summary_csv_path = os.path.join(self.report_dir, 'audio_statistical_summary.csv')
        summary.to_csv(summary_csv_path)
        print(f"Resumen estadístico guardado en {summary_csv_path}")

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(self.pitch_levels, bins=30, kde=True, color='skyblue')
        plt.title('Distribución de Pitch Dominante')
        plt.xlabel('Pitch (Hz)')
        plt.ylabel('Frecuencia')

        plt.subplot(1, 2, 2)
        sns.histplot(self.noise_levels, bins=30, kde=True, color='salmon')
        plt.title('Distribución de Nivel de Ruido (RMS)')
        plt.xlabel('Nivel de Ruido (RMS)')
        plt.ylabel('Frecuencia')

        plt.tight_layout()

        plot_filename = "pitchandrms_visualization.png"
        plot_path = os.path.join(self.images_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Visualización guardada en {plot_path}")

    def visualize_data(self):
        """
        Visualiza formas de onda y espectrogramas de muestras de audio.
        """
        if not self.audios:
            print("No hay audios para visualizar.")
            return

        print("Visualizando audios de muestra...")
        num_samples = min(5, len(self.audios))
        for i in range(num_samples):
            y, sr = self.audios[i]
            file_name = self.file_names[i]
            pitch = self.pitch_levels[i]
            noise = self.noise_levels[i]
            duration = self.durations[i]

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(
                f'{file_name}\nPitch Dominante: {pitch:.2f} Hz\nNivel de Ruido (RMS): {noise:.5f}\nDuración: {duration:.2f} s')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('Amplitud')

            plt.subplot(1, 2, 2)
            X = librosa.stft(y)
            Xdb = librosa.amplitude_to_db(abs(X))
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Espectrograma')
            plt.tight_layout()

            plot_filename = f"{os.path.splitext(file_name)[0]}_visualization.png"
            plot_path = os.path.join(self.images_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Visualización guardada en {plot_path}")

    def generate_report(self):
        """
        Ejecuta todas las funciones necesarias para generar el reporte completo.
        """
        self.load_data()
        if self.audios:
            self.extract_features()
            self.save_results()
            self.statistical_summary()
            self.visualize_data()
        else:
            print("No se encontraron audios para procesar.")

    def evaluate_model(self):
        pass

    def train_model(self):
        pass

    def preprocess_data(self):
        pass
