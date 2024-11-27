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
    """
    AudioProcessor class for processing and analyzing audio files.

    This class provides methods to load audio files from a directory, add noise,
    estimate pitch and noise levels, extract features, save results, generate
    statistical summaries, and visualize audio data.

    Inherits from:
        BaseProcessor (from .base_processor)
    """

    def __init__(self, data_dir, report_dir, noise_factor=0.005):
        """
        Initialize the AudioProcessor.

        Args:
            data_dir (str): Directory containing the audio files.
            report_dir (str): Directory where reports and outputs will be saved.
            noise_factor (float): Factor for injecting noise into the audio signals.
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
        self.durations = []
        self.results_df = pd.DataFrame()

    def load_data(self):
        """
        Load audio files from the data directory.

        Loads audio files with supported extensions, injects noise if specified,
        and stores the audio data along with file names and durations.

        Raises:
            Exception: If an audio file cannot be loaded.
        """
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
        Add Gaussian white noise to an audio signal.

        Args:
            y (ndarray): Original audio signal.

        Returns:
            ndarray: Audio signal with added noise.
        """
        noise = np.random.randn(len(y))
        y_noisy = y + self.noise_factor * noise
        return y_noisy

    def estimate_pitch(self, y, sr):
        """
        Estimate the dominant pitch of an audio signal using librosa.

        Args:
            y (ndarray): Audio signal.
            sr (int): Sampling rate.

        Returns:
            float: Dominant pitch in Hertz (Hz). Returns 0 if no pitch is detected.
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
        Calculate the noise level of an audio signal using RMS (Root Mean Square).

        Args:
            y (ndarray): Audio signal.

        Returns:
            float: Noise level measured as RMS value.
        """
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        return rms_mean

    def extract_features(self):
        """
        Extract features from the loaded audio files.

        For each audio file, estimates the dominant pitch and noise level,
        and stores these values along with file names and durations in a DataFrame.
        """
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
        Save the extracted features to a CSV file.

        Saves the DataFrame containing file names, dominant pitch, noise levels,
        and durations to 'audio_features.csv' in the report directory.
        """
        csv_path = os.path.join(self.report_dir, 'audio_features.csv')
        self.results_df.to_csv(csv_path, index=False)
        print(f"Resultados guardados en {csv_path}")

    def statistical_summary(self):
        """
        Generate basic statistical summaries of the audio features.

        Creates descriptive statistics of the extracted features, saves them to a CSV file,
        and generates visualizations of pitch and noise level distributions.
        """
        print("Generando resúmenes estadísticos de las características de audio...")
        summary = self.results_df.describe()
        summary_csv_path = os.path.join(self.report_dir, 'audio_statistical_summary.csv')
        summary.to_csv(summary_csv_path)
        print(f"Resumen estadístico guardado en {summary_csv_path}")

        # Generar visualizaciones
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
        Visualize waveforms and spectrograms of sample audio files.

        For a subset of audio files, plots the waveform and spectrogram,
        including annotations of pitch, noise level, and duration.
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
        Execute all necessary functions to generate the complete audio report.

        Runs the full processing pipeline: loading data, extracting features,
        saving results, generating statistical summaries, and visualizing data.
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
        """
        Placeholder method for evaluating a model.

        Currently not implemented.
        """
        pass

    def train_model(self):
        """
        Placeholder method for training a model.

        Currently not implemented.
        """
        pass

    def preprocess_data(self):
        """
        Placeholder method for preprocessing data.

        Currently not implemented.
        """
        pass
