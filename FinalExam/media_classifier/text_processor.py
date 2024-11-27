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
    """
    TextProcessor class for processing and analyzing text documents.

    This class provides methods to load text documents from a directory, preprocess them,
    extract word frequencies and keywords, save results, generate statistical summaries,
    and visualize data such as word frequencies and word clouds.

    Inherits from:
        BaseProcessor (from .base_processor)

    Attributes:
        data_dir (str): Directory containing the text files.
        report_dir (str): Directory where reports and outputs will be saved.
        documents (list): List of raw text documents.
        file_names (list): List of filenames corresponding to the documents.
        images_dir (str): Directory for saving image outputs.
        processed_documents (list): List of preprocessed text documents.
        word_frequencies (Counter): Counter object with word frequencies.
        tfidf_matrix (sparse matrix): TF-IDF matrix of the processed documents.
        keywords (dict): Dictionary mapping filenames to their extracted keywords.
        stop_words (set): Set of stopwords to exclude during processing.
        nlp (spacy.lang): SpaCy language model for lemmatization.
    """

    def __init__(self, data_dir, report_dir, language='english', custom_stopwords=None):
        """
        Initialize the TextProcessor.

        Args:
            data_dir (str): Directory containing the text files.
            report_dir (str): Directory where reports and outputs will be saved.
            language (str): Language of the texts for stopword processing.
            custom_stopwords (list, optional): Additional words to exclude from analysis.
        """
        self.data_dir = data_dir
        self.report_dir = report_dir
        ensure_directory(self.report_dir)
        self.documents = []
        self.file_names = []
        self.images_dir = os.path.join(self.report_dir, 'images', 'text_processor')
        self.processed_documents = []
        self.word_frequencies = None
        self.tfidf_matrix = None
        self.keywords = {}
        self.stop_words = set(stopwords.words(language))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.max_length = 2000000

    def load_data(self):
        """
        Load text documents from the data directory.

        Loads text files with supported extensions, reads their content,
        and stores the documents along with their filenames.

        Raises:
            Exception: If a file cannot be read.
        """
        print(f"Cargando documentos de texto desde {self.data_dir}...")
        text_extensions = ('.txt', '.csv', '.md', '.json')
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
        Split a list of tokens into fragments not exceeding max_length characters.

        Args:
            tokens (list): List of tokens.
            max_length (int): Maximum number of characters per fragment.

        Returns:
            list: List of token fragments.
        """
        fragments = []
        current_fragment = []
        current_length = 0

        for token in tokens:
            token_length = len(token) + 1  # Adding 1 for space
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
        """
        Preprocess the text documents.

        Performs tokenization, lowercasing, stopword removal, lemmatization,
        and handles large texts by splitting them into fragments.
        """
        print("Preprocesando documentos...")
        for idx, doc in enumerate(self.documents):
            # Tokenization and basic cleaning
            tokens = word_tokenize(doc.lower())
            # Filter out non-alphabetic tokens and remove stopwords
            tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]

            # Split tokens into fragments if they exceed SpaCy's max length
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
        """
        Extract word frequencies from the processed documents.

        Combines all processed documents into one text, splits into words,
        and counts the frequency of each word.
        """
        print("Extrayendo frecuencias de palabras...")
        all_words = ' '.join(self.processed_documents).split()
        self.word_frequencies = Counter(all_words)
        print("Frecuencias de palabras extraídas.")

    def extract_keywords(self, top_n=10):
        """
        Extract keywords using TF-IDF for each document.

        Args:
            top_n (int): Number of top keywords to extract per document.
        """
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
        """
        Save the extracted word frequencies and keywords to CSV files.

        Saves global word frequencies, keywords per document, and word frequencies per document.
        """
        print("Guardando resultados en archivos CSV...")
        # Save word frequencies
        freq_df = pd.DataFrame(self.word_frequencies.items(), columns=['word', 'frequency'])
        freq_csv_path = os.path.join(self.report_dir, 'word_frequencies.csv')
        freq_df.to_csv(freq_csv_path, index=False)
        print(f"Frecuencias de palabras guardadas en {freq_csv_path}")

        # Save keywords
        keywords_list = []
        for file, kws in self.keywords.items():
            for word, score in kws:
                keywords_list.append({'file_name': file, 'keyword': word, 'tfidf_score': score})
        keywords_df = pd.DataFrame(keywords_list)
        keywords_csv_path = os.path.join(self.report_dir, 'keywords.csv')
        keywords_df.to_csv(keywords_csv_path, index=False)
        print(f"Palabras clave guardadas en {keywords_csv_path}")

        # Save word frequencies per document
        print("Guardando frecuencias de palabras por documento en archivos CSV...")
        for idx, doc in enumerate(self.processed_documents):
            word_counts = Counter(doc.split())
            doc_freq_df = pd.DataFrame(word_counts.items(), columns=['word', 'frequency'])
            # Replace invalid characters in filename
            safe_file_name = ''.join(c for c in self.file_names[idx] if c.isalnum() or c in (' ', '_')).rstrip()
            doc_freq_csv_path = os.path.join(self.report_dir, f'word_frequencies_{safe_file_name}.csv')
            doc_freq_df.to_csv(doc_freq_csv_path, index=False)
            print(f"Frecuencias de palabras para {self.file_names[idx]} guardadas en {doc_freq_csv_path}")

    def visualize_word_frequencies(self, top_n=20):
        """
        Visualize the top N most frequent words globally and per document.

        Args:
            top_n (int): Number of top words to display in the visualization.
        """
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

        plot_filename = f'Top_{top_n}_Palabras_Mas_Frecuentes.png'
        plot_path = os.path.join(self.images_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Visualización guardada en {plot_path}")

        # Visualize top N words per document
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

            plot_filename = f'Top_{top_n}_Palabras_Mas_Frecuentes_en_{self.file_names[idx]}.png'
            plot_path = os.path.join(self.images_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Visualización guardada en {plot_path}")

    def visualize_wordcloud(self):
        """
        Generate and save word clouds for the global corpus and per document.
        """
        print("Generando nube de palabras global...")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
            self.word_frequencies)
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nube de Palabras Global')

        plot_filename = 'wordcloud_global.png'
        plot_path = os.path.join(self.images_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Visualización guardada en {plot_path}")

        # Generate word clouds per document
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

            plot_filename = f'wordcloud_{self.file_names[idx]}.png'
            plot_path = os.path.join(self.images_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Visualización guardada en {plot_path}")

    def visualize_keywords(self, top_n=10):
        """
        Visualize keywords extracted from each document.

        Args:
            top_n (int): Number of top keywords to display.
        """
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

            plot_filename = f'keywords_{file}.png'
            plot_path = os.path.join(self.images_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"Visualización guardada en {plot_path}")

    def statistical_summary(self):
        """
        Generate basic statistical summaries of the processed text.

        Creates summaries such as word counts per document, global statistics,
        and visualizes the distribution of word counts.
        """
        print("Generando resúmenes estadísticos de los textos...")
        # Word counts per document
        word_counts = [len(doc.split()) for doc in self.processed_documents]
        word_counts_df = pd.DataFrame({'document_name': self.file_names, 'word_count': word_counts})
        word_counts_csv_path = os.path.join(self.report_dir, 'text_word_counts_summary.csv')
        word_counts_df.to_csv(word_counts_csv_path, index=False)
        print(f"Resumen de conteo de palabras por documento guardado en {word_counts_csv_path}")

        # Global statistics
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

        # Visualize word count distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(word_counts, bins=30, kde=True, color='purple')
        plt.title('Distribución de la Cantidad de Palabras por Documento')
        plt.xlabel('Cantidad de Palabras')
        plt.ylabel('Frecuencia')

        plot_filename = 'word_count_distribution.png'
        plot_path = os.path.join(self.images_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Visualización guardada en {plot_path}")

    def generate_report(self):
        """
        Execute all functions to generate the complete text processing report.

        Runs the full processing pipeline: loading data, preprocessing text,
        extracting word frequencies and keywords, saving results, generating
        statistical summaries, and visualizing data.
        """
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

    # Inherited methods not used in this text processor
    def train_model(self):
        """
        Placeholder method for training a model.

        Currently not implemented.
        """
        pass

    def evaluate_model(self):
        """
        Placeholder method for evaluating a model.

        Currently not implemented.
        """
        pass

    def visualize_data(self):
        """
        Placeholder method for visualizing data.

        Currently not implemented.
        """
        pass

    def preprocess_data(self):
        """
        Placeholder method for preprocessing data.

        Currently not implemented.
        """
        pass

    def extract_features(self):
        """
        Placeholder method for extracting features.

        Currently not implemented.
        """
        pass
