"""
Preprocesamiento y Análisis de Textos de Artículos de Investigación.

Este codigo realiza las siguientes tareas:
1. Extrae texto de archivos PDF en una carpeta específica.
2. Preprocesa el texto (convierte a minúsculas, elimina números y puntuación).
3. Tokeniza el texto y elimina palabras vacías.
4. Lematiza los tokens.
5. Calcula los términos más frecuentes, incluyendo bigramas y trigramas.
6. Realiza análisis de tópicos utilizando LDA.
7. Extrae entidades nombradas.
8. Extrae secciones específicas (Resumen y Referencias) de los textos.
9. Visualiza los términos más frecuentes y genera una nube de palabras avanzada.
10. Evalúa la similitud entre documentos y realiza clustering.
11. Genera reportes automatizados.
"""

import os
import re
import nltk
import PyPDF2
import matplotlib.pyplot as plt
import spacy
import gensim
import pandas as pd
import warnings

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from wordcloud import WordCloud
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")  # Para ignorar avisos de depreciación

def descargar_recursos():
    """
    Descarga los recursos necesarios de NLTK y spaCy.
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    spacy.cli.download('es_core_news_sm')

def extraer_texto_de_pdfs(ruta_carpeta):
    """
    Extrae el texto de todos los archivos PDF en la carpeta especificada.

    Parámetros:
    ruta_carpeta (str): Ruta de la carpeta que contiene los archivos PDF.

    Retorna:
    list: Lista de strings con el texto extraído de cada PDF.
    """
    textos = []
    for archivo in os.listdir(ruta_carpeta):
        if archivo.lower().endswith('.pdf'):
            ruta_pdf = os.path.join(ruta_carpeta, archivo)
            with open(ruta_pdf, 'rb') as f:
                lector_pdf = PyPDF2.PdfReader(f)
                texto = ''
                for pagina in lector_pdf.pages:
                    texto_pagina = pagina.extract_text()
                    if texto_pagina:
                        texto += texto_pagina
                textos.append(texto)
    return textos

def preprocesar_texto(texto):
    """
    Preprocesa el texto: convierte a minúsculas, elimina números y puntuación, y tokeniza.

    Parámetros:
    texto (str): Texto a preprocesar.

    Retorna:
    list: Lista de tokens.
    """
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)  # Elimina números
    texto = re.sub(r'[^\w\s]', '', texto)  # Elimina puntuación
    tokens = word_tokenize(texto, language='spanish')
    return tokens

def eliminar_stopwords(tokens):
    """
    Elimina las palabras vacías en español e inglés, y palabras personalizadas no deseadas.

    Parámetros:
    tokens (list): Lista de tokens.

    Retorna:
    list: Lista de tokens filtrados.
    """
    stop_words_spanish = set(stopwords.words('spanish'))
    stop_words_english = set(stopwords.words('english'))
    stop_words = stop_words_spanish.union(stop_words_english)

    # Lista de palabras adicionales a eliminar
    palabras_no_deseadas = {
        'et', 'al', 'doi', 'httpsdoiorgedutec', 'issn', 'edutec', 'revista',
        'electrónico', 'página', 'http', 'https', 'www', 'com', 'org', 'vol',
        'pp', 'isbn', 'número', 'fig', 'figure', 'table', 'author', 'authors',
        'et', 'sti', 'n'
    }
    stop_words.update(palabras_no_deseadas)

    tokens_filtrados = [token for token in tokens if token not in stop_words]
    return tokens_filtrados


def lematizar_tokens(tokens):
    """
    Lematiza una lista de tokens utilizando spaCy.

    Parámetros:
    tokens (list): Lista de tokens.

    Retorna:
    list: Lista de tokens lematizados.
    """
    nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
    doc = nlp(' '.join(tokens))
    tokens_lematizados = [token.lemma_ for token in doc]
    return tokens_lematizados

def extraer_terminos_frecuentes(tokens, n=20):
    """
    Extrae los 'n' términos más frecuentes de una lista de tokens.

    Parámetros:
    tokens (list): Lista de tokens.
    n (int): Número de términos frecuentes a extraer.

    Retorna:
    list: Lista de tuplas (término, frecuencia).
    """
    freq_dist = FreqDist(tokens)
    terminos_comunes = freq_dist.most_common(n)
    return terminos_comunes

def extraer_ngramas(tokens, ngram=2, top_n=10):
    """
    Extrae los n-gramas más frecuentes de una lista de tokens.

    Parámetros:
    tokens (list): Lista de tokens.
    ngram (int): Tamaño del n-grama (2 para bigramas, 3 para trigramas).
    top_n (int): Número de n-gramas a extraer.

    Retorna:
    list: Lista de tuplas (n-grama, frecuencia).
    """
    if ngram == 2:
        ngramas = list(bigrams(tokens))
    elif ngram == 3:
        ngramas = list(trigrams(tokens))
    else:
        raise ValueError("El valor de ngram debe ser 2 o 3.")
    freq_dist = FreqDist(ngramas)
    ngramas_comunes = freq_dist.most_common(top_n)
    return ngramas_comunes

def analizar_colocaciones(tokens, top_n=10):
    """
    Analiza las colocaciones más significativas en una lista de tokens.

    Parámetros:
    tokens (list): Lista de tokens.
    top_n (int): Número de colocaciones a mostrar.

    Retorna:
    list: Lista de tuplas (colocación, puntuación).
    """
    finder = BigramCollocationFinder.from_words(tokens)
    bigramas_puntuados = finder.score_ngrams(BigramAssocMeasures.likelihood_ratio)
    top_bigrams = bigramas_puntuados[:top_n]
    return top_bigrams


def extraer_seccion_mejorada(texto, seccion):
    """
    Extrae una sección específica del texto entre dos delimitadores, manejando variaciones en los títulos.

    Parámetros:
    texto (str): Texto completo.
    seccion (str): Nombre de la sección a extraer.

    Retorna:
    str: Texto de la sección extraída o None si no se encuentra.
    """
    patrones = {
        'resumen': r'(?:resumen|abstract)',
        'introducción': r'(?:introducción|introduction)',
        'metodología': r'(?:metodología|methodology)',
        'resultados': r'(?:resultados|results)',
        'conclusiones': r'(?:conclusiones|conclusion)',
        'referencias': r'(?:referencias|bibliografía|references)'
    }
    inicio_patron = patrones.get(seccion.lower(), seccion)

    # Excluir el patrón de inicio del patrón de fin
    patrones_sin_inicio = patrones.copy()
    del patrones_sin_inicio[seccion.lower()]
    fin_patron = '|'.join(patrones_sin_inicio.values())

    # Si no hay más secciones, capturar hasta el final del texto
    if not fin_patron:
        fin_patron = r'\Z'

    # Construir la expresión regular sin flags inline
    patron = rf"{inicio_patron}\s*(.*?)(?={fin_patron})"

    # Usar flags en re.findall()
    seccion_encontrada = re.findall(patron, texto, flags=re.DOTALL | re.IGNORECASE)
    if seccion_encontrada:
        return seccion_encontrada[0].strip()
    else:
        return None


def extraer_entidades(texto):
    """
    Extrae entidades nombradas del texto utilizando spaCy.

    Parámetros:
    texto (str): Texto del cual extraer entidades.

    Retorna:
    list: Lista de tuplas (entidad, etiqueta).
    """
    nlp = spacy.load('es_core_news_sm')
    doc = nlp(texto)
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    return entidades

def analizar_topicos(textos, num_topicos=5):
    """
    Realiza análisis de tópicos utilizando LDA sobre una lista de textos.

    Parámetros:
    textos (list): Lista de textos.
    num_topicos (int): Número de tópicos a extraer.

    Retorna:
    LdaModel: Modelo LDA entrenado.
    """
    # Preprocesamiento para LDA
    textos_tokenizados = [text.split() for text in textos]
    dictionary = corpora.Dictionary(textos_tokenizados)
    corpus = [dictionary.doc2bow(texto) for texto in textos_tokenizados]

    # Entrenar modelo LDA
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topicos, passes=10)
    return lda_model, corpus, dictionary

def visualizar_topicos(lda_model, dictionary):
    """
    Visualiza los tópicos generados por el modelo LDA.

    Parámetros:
    lda_model (LdaModel): Modelo LDA entrenado.
    dictionary (Dictionary): Diccionario de términos.
    """
    for idx, topic in lda_model.print_topics(-1):
        print(f"Tópico {idx+1}: {topic}")

def calcular_similitud(textos):
    """
    Calcula la similitud entre documentos utilizando TF-IDF y similitud de coseno.

    Parámetros:
    textos (list): Lista de textos.

    Retorna:
    numpy.ndarray: Matriz de similitud.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(textos)
    similitudes = cosine_similarity(tfidf_matrix)
    return similitudes, tfidf_matrix

def clustering_documentos(tfidf_matrix, num_clusters=5):
    """
    Realiza clustering de documentos utilizando KMeans.

    Parámetros:
    tfidf_matrix (sparse matrix): Matriz TF-IDF de los textos.
    num_clusters (int): Número de clusters.

    Retorna:
    list: Lista de etiquetas de cluster para cada documento.
    """
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    return clusters

def visualizar_terminos_frecuentes(terminos_comunes):
    """
    Visualiza los términos más frecuentes mediante un gráfico de barras.

    Parámetros:
    terminos_comunes (list): Lista de tuplas (término, frecuencia).
    """
    terminos = [termino for termino, frecuencia in terminos_comunes]
    frecuencias = [frecuencia for termino, frecuencia in terminos_comunes]

    plt.figure(figsize=(10, 5))
    plt.bar(terminos, frecuencias, color='skyblue')
    plt.xticks(rotation=45)
    plt.title('Términos Más Frecuentes')
    plt.xlabel('Términos')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

def generar_nube_palabras(tokens):
    """
    Genera y muestra una nube de palabras a partir de una lista de tokens.

    Parámetros:
    tokens (list): Lista de tokens.
    """
    texto = ' '.join(tokens)
    nube_palabras = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(texto)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(nube_palabras, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def generar_reporte(terminos_comunes, bigramas_comunes, trigramas_comunes, topicos, entidades, clusters, similitudes):
    """
    Genera un reporte consolidado de los resultados.

    Parámetros:
    terminos_comunes (list): Términos más frecuentes.
    bigramas_comunes (list): Bigramas más frecuentes.
    trigramas_comunes (list): Trigramas más frecuentes.
    topicos (LdaModel): Modelo LDA con tópicos.
    entidades (list): Entidades nombradas extraídas.
    clusters (list): Etiquetas de cluster para cada documento.
    similitudes (numpy.ndarray): Matriz de similitud entre documentos.
    """
    # Creación de un DataFrame para el reporte
    reporte = pd.DataFrame({
        'Términos Frecuentes': [terminos_comunes],
        'Bigramas Comunes': [bigramas_comunes],
        'Trigramas Comunes': [trigramas_comunes],
        'Entidades Nombradas': [entidades],
        'Clusters': [clusters],
        'Similitud entre Documentos': [similitudes]
    })

    # Guardar el reporte en un archivo Excel
    reporte.to_excel('reporte_analisis.xlsx', index=False)
    print("Reporte generado: 'reporte_analisis.xlsx'")

def main():
    """
    Función principal que coordina la ejecución del script.
    """
    # Descargar recursos necesarios
    descargar_recursos()

    # Definir la ruta a la carpeta con los PDFs
    ruta_carpeta = './assets/'  # Reemplaza con la ruta correcta

    # Extraer texto de los PDFs
    print("Extrayendo texto de los PDFs...")
    textos = extraer_texto_de_pdfs(ruta_carpeta)

    # Preprocesar y tokenizar textos
    print("Preprocesando textos...")
    tokens_total = []
    textos_procesados = []  # Lista para LDA y similitud
    for texto in textos:
        tokens = preprocesar_texto(texto)
        tokens = eliminar_stopwords(tokens)
        tokens = lematizar_tokens(tokens)
        tokens_total.extend(tokens)
        textos_procesados.append(' '.join(tokens))

    # Extraer términos más frecuentes
    print("Calculando términos más frecuentes...")
    terminos_comunes = extraer_terminos_frecuentes(tokens_total, n=20)
    print("Términos más frecuentes:")
    for termino, frecuencia in terminos_comunes:
        print(f"{termino}: {frecuencia}")

    # Extraer bigramas y trigramas más frecuentes
    print("Extrayendo bigramas y trigramas más frecuentes...")
    bigramas_comunes = extraer_ngramas(tokens_total, ngram=2, top_n=10)
    trigramas_comunes = extraer_ngramas(tokens_total, ngram=3, top_n=10)

    print("\nBigramas más comunes:")
    for bigrama, frecuencia in bigramas_comunes:
        print(f"{bigrama}: {frecuencia}")

    print("\nTrigramas más comunes:")
    for trigrama, frecuencia in trigramas_comunes:
        print(f"{trigrama}: {frecuencia}")

    # Análisis de colocaciones
    print("\nAnalizando colocaciones más significativas...")
    colocaciones = analizar_colocaciones(tokens_total, top_n=10)
    print("Colocaciones más significativas:")
    for colocacion, puntuacion in colocaciones:
        print(f"{colocacion}: {puntuacion}")

    # Análisis de tópicos con LDA
    print("\nRealizando análisis de tópicos...")
    lda_model, corpus, dictionary = analizar_topicos(textos_procesados, num_topicos=5)
    visualizar_topicos(lda_model, dictionary)

    # Extracción de entidades nombradas
    print("\nExtrayendo entidades nombradas...")
    entidades = []
    for texto in textos_procesados:
        entidades.extend(extraer_entidades(texto))
    print("Entidades nombradas extraídas:")
    print(entidades)

    # Extraer secciones específicas
    print("\nExtrayendo secciones 'Resumen' y 'Referencias'...")
    resumenes = []
    referencias = []
    for texto in textos:
        resumen = extraer_seccion_mejorada(texto, 'resumen')
        if resumen:
            resumenes.append(resumen)
        referencia = extraer_seccion_mejorada(texto, 'referencias')
        if referencia:
            referencias.append(referencia)

    # Visualizar términos más frecuentes
    print("\nVisualizando términos más frecuentes...")
    visualizar_terminos_frecuentes(terminos_comunes)

    # Generar nube de palabras
    print("\nGenerando nube de palabras...")
    generar_nube_palabras(tokens_total)

    # Calcular similitud entre documentos
    print("\nCalculando similitud entre documentos...")
    similitudes, tfidf_matrix = calcular_similitud(textos_procesados)
    print("Matriz de similitud:")
    print(similitudes)

    # Clustering de documentos
    print("\nRealizando clustering de documentos...")
    clusters = clustering_documentos(tfidf_matrix, num_clusters=3)
    print("Etiquetas de cluster para cada documento:")
    print(clusters)

    # Generar reporte
    print("\nGenerando reporte...")
    generar_reporte(terminos_comunes, bigramas_comunes, trigramas_comunes, lda_model, entidades, clusters, similitudes)

if __name__ == "__main__":
    main()
