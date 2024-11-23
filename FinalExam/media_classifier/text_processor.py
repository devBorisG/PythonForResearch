# media_classifier/text_processor.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from .base_processor import BaseProcessor
from .utils import ensure_directory

# Descargar recursos de NLTK si no están disponibles
nltk.download('punkt')
nltk.download('stopwords')


class TextProcessor(BaseProcessor, ABC):
    def __init__(self, data_dir, report_dir, language='english', custom_stopwords=None):
        """
        Inicializa el procesador de texto.

        :param data_dir: Directorio que contiene los archivos de texto.
        :param report_dir: Directorio donde se guardarán los reportes y resultados.
        :param language: Idioma de los textos para el procesamiento de stopwords.
        :param custom_stopwords: Lista de palabras adicionales a excluir del análisis.
        """
        self.data_dir = data_dir
        self.report_dir = report_dir
        ensure_directory(self.report_dir)
        self.documents = []
        self.file_names = []
        self.processed_documents = []
        self.word_frequencies = None
        self.tfidf_matrix = None
        self.keywords = {}
        self.stop_words = set(stopwords.words(language))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        self.nlp = spacy.load('en_core_web_sm')  # Cambia el modelo según el idioma
        self.nlp.max_length = 2000000  # Opcional: Aumentar el límite de SpaCy

    def load_data(self):
        print(f"Cargando documentos de texto desde {self.data_dir}...")
        text_extensions = ('.txt', '.csv', '.md', '.json')  # Puedes ampliar según tus necesidades
        text_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(text_extensions)]

        if not text_files:
            print("No se encontraron archivos de texto en el directorio especificado.")
            return

        for file in text_files:
            file_path = os.path.join(self.data_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.documents.append(content)
                    self.file_names.append(file)
                    print(f"Documento {file_path} cargado exitosamente.")
            except Exception as e:
                print(f"Error al cargar el documento {file_path}: {e}")
        print(f"Se cargaron {len(self.documents)} documentos en total.")

    def split_text(self, tokens, max_length=1000000):
        """
        Divide una lista de tokens en fragmentos que no excedan max_length caracteres.

        :param tokens: Lista de tokens.
        :param max_length: Número máximo de caracteres por fragmento.
        :return: Lista de fragmentos de tokens.
        """
        fragments = []
        current_fragment = []
        current_length = 0

        for token in tokens:
            token_length = len(token) + 1  # +1 para el espacio
            if current_length + token_length > max_length:
                if current_fragment:
                    fragments.append(current_fragment)
                current_fragment = [token]
                current_length = token_length
            else:
                current_fragment.append(token)
                current_length += token_length

        if current_fragment:
            fragments.append(current_fragment)

        return fragments

    def preprocess_text(self):
        print("Preprocesando documentos...")
        for idx, doc in enumerate(self.documents):
            # Tokenización y limpieza básica
            tokens = word_tokenize(doc.lower())
            # Filtrar tokens que no son alfabéticos y eliminar stopwords
            tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]

            # Dividir tokens en fragmentos si exceden el límite de SpaCy
            fragments = self.split_text(tokens, max_length=self.nlp.max_length)

            lemmatized = []
            for fragment in fragments:
                try:
                    doc_spacy = self.nlp(' '.join(fragment))
                    lemmatized.extend([token.lemma_ for token in doc_spacy])
                except Exception as e:
                    print(f"Error al procesar fragmento {idx}: {e}")

            self.processed_documents.append(' '.join(lemmatized))
            print(f"Documento {self.file_names[idx]} preprocesado.")
        print("Preprocesamiento completado.")

    def extract_word_frequencies(self):
        print("Extrayendo frecuencias de palabras...")
        all_words = ' '.join(self.processed_documents).split()
        self.word_frequencies = Counter(all_words)
        print("Frecuencias de palabras extraídas.")

    def extract_keywords(self, top_n=10):
        print("Extrayendo palabras clave utilizando TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = vectorizer.fit_transform(self.processed_documents)
        feature_names = vectorizer.get_feature_names_out()

        for idx, row in enumerate(self.tfidf_matrix):
            tfidf_scores = row.toarray()[0]
            top_indices = tfidf_scores.argsort()[-top_n:][::-1]
            top_keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices]
            self.keywords[self.file_names[idx]] = top_keywords
            print(f"Palabras clave para {self.file_names[idx]} extraídas.")
        print("Extracción de palabras clave completada.")

    def save_results(self):
        print("Guardando resultados en archivos CSV...")
        # Guardar frecuencias de palabras
        freq_df = pd.DataFrame(self.word_frequencies.items(), columns=['word', 'frequency'])
        freq_csv_path = os.path.join(self.report_dir, 'word_frequencies.csv')
        freq_df.to_csv(freq_csv_path, index=False)
        print(f"Frecuencias de palabras guardadas en {freq_csv_path}")

        # Guardar palabras clave
        keywords_list = []
        for file, kws in self.keywords.items():
            for word, score in kws:
                keywords_list.append({'file_name': file, 'keyword': word, 'tfidf_score': score})
        keywords_df = pd.DataFrame(keywords_list)
        keywords_csv_path = os.path.join(self.report_dir, 'keywords.csv')
        keywords_df.to_csv(keywords_csv_path, index=False)
        print(f"Palabras clave guardadas en {keywords_csv_path}")

        # Guardar frecuencias de palabras por documento
        print("Guardando frecuencias de palabras por documento en archivos CSV...")
        for idx, doc in enumerate(self.processed_documents):
            word_counts = Counter(doc.split())
            doc_freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'frequency'])
            # Reemplazar caracteres no válidos en el nombre del archivo
            safe_file_name = ''.join(c for c in self.file_names[idx] if c.isalnum() or c in (' ', '_')).rstrip()
            doc_freq_csv_path = os.path.join(self.report_dir, f'word_frequencies_{safe_file_name}.csv')
            doc_freq_df.to_csv(doc_freq_csv_path, index=False)
            print(f"Frecuencias de palabras para {self.file_names[idx]} guardadas en {doc_freq_csv_path}")

    def visualize_word_frequencies(self, top_n=20):
        print(f"Visualizando las {top_n} palabras más frecuentes globalmente...")
        most_common = self.word_frequencies.most_common(top_n)
        words, counts = zip(*most_common)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(words), y=list(counts), palette='viridis')
        plt.title(f'Top {top_n} Palabras Más Frecuentes')
        plt.xlabel('Palabra')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Visualizar top_n palabras por documento
        print(f"Visualizando las {top_n} palabras más frecuentes por documento...")
        for idx, doc in enumerate(self.processed_documents):
            word_counts = Counter(doc.split())
            most_common_doc = word_counts.most_common(top_n)
            if not most_common_doc:
                print(f"No hay palabras suficientes para {self.file_names[idx]}.")
                continue
            words_doc, counts_doc = zip(*most_common_doc)

            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(words_doc), y=list(counts_doc), palette='viridis')
            plt.title(f'Top {top_n} Palabras Más Frecuentes en {self.file_names[idx]}')
            plt.xlabel('Palabra')
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def visualize_wordcloud(self):
        print("Generando nube de palabras global...")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
            self.word_frequencies)
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nube de Palabras Global')
        plt.show()

        # Generar nube de palabras por documento
        print("Generando nubes de palabras por documento...")
        for idx, doc in enumerate(self.processed_documents):
            word_counts = Counter(doc.split())
            if not word_counts:
                print(f"No hay palabras para generar la nube de palabras de {self.file_names[idx]}.")
                continue
            wordcloud_doc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
                word_counts)
            plt.figure(figsize=(15, 7.5))
            plt.imshow(wordcloud_doc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Nube de Palabras para {self.file_names[idx]}')
            plt.show()

    def visualize_keywords(self, top_n=10):
        print("Visualizando palabras clave por documento...")
        for file, kws in self.keywords.items():
            if not kws:
                print(f"No hay palabras clave para {file}.")
                continue
            words, scores = zip(*kws)
            plt.figure(figsize=(10, 5))
            sns.barplot(x=list(words), y=list(scores), palette='magma')
            plt.title(f'Palabras Clave para {file}')
            plt.xlabel('Palabra Clave')
            plt.ylabel('Puntuación TF-IDF')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def statistical_summary(self):
        """
        Genera resúmenes estadísticos básicos del texto procesado.
        """
        print("Generando resúmenes estadísticos de los textos...")
        # Número de palabras por documento
        word_counts = [len(doc.split()) for doc in self.processed_documents]
        word_counts_df = pd.DataFrame({'document_name': self.file_names, 'word_count': word_counts})
        word_counts_csv_path = os.path.join(self.report_dir, 'text_word_counts_summary.csv')
        word_counts_df.to_csv(word_counts_csv_path, index=False)
        print(f"Resumen de conteo de palabras por documento guardado en {word_counts_csv_path}")

        # Estadísticas globales
        total_words = sum(word_counts)
        avg_words = np.mean(word_counts)
        max_words = np.max(word_counts)
        min_words = np.min(word_counts)
        global_stats = {
            'Total de Palabras': [total_words],
            'Promedio de Palabras por Documento': [avg_words],
            'Máximo de Palabras en un Documento': [max_words],
            'Mínimo de Palabras en un Documento': [min_words]
        }
        global_stats_df = pd.DataFrame(global_stats)
        global_stats_csv_path = os.path.join(self.report_dir, 'text_global_stats_summary.csv')
        global_stats_df.to_csv(global_stats_csv_path, index=False)
        print(f"Resumen estadístico global guardado en {global_stats_csv_path}")

        # Visualización de distribución de palabras por documento
        plt.figure(figsize=(12, 6))
        sns.histplot(word_counts, bins=30, kde=True, color='purple')
        plt.title('Distribución de la Cantidad de Palabras por Documento')
        plt.xlabel('Cantidad de Palabras')
        plt.ylabel('Frecuencia')
        plt.show()

    def generate_report(self):
        print("Generando reporte completo...")
        self.load_data()
        if self.documents:
            self.preprocess_text()
            self.extract_word_frequencies()
            self.extract_keywords()
            self.save_results()
            self.statistical_summary()
            self.visualize_word_frequencies()
            self.visualize_wordcloud()
            self.visualize_keywords()
            print("=== Reporte generado exitosamente ===")
        else:
            print("No se encontraron documentos para procesar.")

    # Métodos heredados que no se utilizan en este procesador de texto
    def train_model(self):
        pass

    def evaluate_model(self):
        pass

    def visualize_data(self):
        pass

    def preprocess_data(self):
        pass

    def extract_features(self):
        pass
