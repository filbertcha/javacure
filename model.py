import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from Levenshtein import distance as levenshtein_distance
import random

def levenshtein_correction(word, dataset):
    """
    Mengembalikan kata terdekat berdasarkan Levenshtein Distance.
    """
    # Pastikan dataset dapat diiterasi
    if isinstance(dataset, pd.Series):  # Jika dataset adalah Series
        dataset = dataset.tolist()  # Konversi ke list

    if word in dataset:  # Jika kata sudah ada di dataset
        return word

    # Hitung jarak Levenshtein untuk setiap kata dalam dataset
    distances = [levenshtein_distance(word, w) for w in dataset]
    corrected_word = dataset[np.argmin(distances)]
    return corrected_word

def kmeans_correction(word, dataset, kmeans_model, tfidf_vectorizer):
    """
    Mengoreksi kata menggunakan model K-Means.
    """
    word_vector = tfidf_vectorizer.transform([word])
    cluster_label = kmeans_model.predict(word_vector)[0]
    cluster_indices = np.where(kmeans_model.labels_ == cluster_label)[0]
    cluster_words = [dataset.iloc[i] for i in cluster_indices]

    # Fallback jika tidak ada cluster
    if not cluster_words:
        return word  # Tidak mengubah kata jika cluster kosong

    return levenshtein_correction(word, cluster_words)

def evaluate_model_with_confusion_matrix(model_func, dataset, test_words, true_corrections, tfidf_vectorizer=None, model=None):
    """
    Evaluasi model dengan custom confusion matrix sesuai definisi:
    - True Positive: Model tidak mengubah kata, sesuai dengan label.
    - True Negative: Model mengubah kata sesuai dengan label.
    - False Positive: Model tidak mengubah kata, tetapi salah.
    - False Negative: Model mengubah kata, tetapi salah.
    """
    # Inisialisasi counter untuk tiap kategori
    confusion_matrix = np.zeros(4, dtype=int)

    for word, true_label in zip(test_words, true_corrections):
        # Prediksi model
        if model_func == levenshtein_correction:
            prediction = model_func(word, dataset)
        else:
            prediction = model_func(word, dataset, model, tfidf_vectorizer)

        # Penentuan prediksi
        if word == prediction:  # Model tidak melakukan prediksi
            if word == true_label:
                confusion_matrix[0] += 1  # True Positive
            else:
                confusion_matrix[2] += 1  # False Positive
        else:  # Model melakukan prediksi
            if prediction == true_label:
                confusion_matrix[3] += 1  # True Negative
            else:
                confusion_matrix[1] += 1  # False Negative

    # Confusion matrix
    return confusion_matrix

def generate_typo(word, max_typos=2):
    """
    Menghasilkan kata salah (typo) dari input kata benar.
    Args:
        word (str): Kata benar.
        max_typos (int): Jumlah maksimal kesalahan.
    Returns:
        str: Kata salah yang dihasilkan.
    """
    word = list(word)
    for _ in range(random.randint(1, max_typos)):
        typo_type = random.choice(['delete', 'swap', 'replace', 'insert'])
        if typo_type == 'delete' and len(word) > 1:
            idx = random.randint(0, len(word) - 1)
            word.pop(idx)
        elif typo_type == 'swap' and len(word) > 1:
            idx = random.randint(0, len(word) - 2)
            word[idx], word[idx + 1] = word[idx + 1], word[idx]
        elif typo_type == 'replace':
            idx = random.randint(0, len(word) - 1)
            word[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
        elif typo_type == 'insert':
            idx = random.randint(0, len(word))
            word.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz'))
    return ''.join(word)