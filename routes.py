


from flask import Flask, render_template, request, jsonify
import model
import pandas as pd
import os
from __init__ import app

# Initialize model and dataset
@app.before_request
def initialize():
    global X, kmeans, tfidf_vectorizer, accuracy_metrics
    
    # Load dataset
    try:
        file_path = 'data_preprocessing_sastra_jawa_full.xlsx'
        sastra_df = pd.read_excel(file_path)
        X = sastra_df['Kalimat'].dropna()
        X = X.astype(str)
    except FileNotFoundError:
        # For testing without the dataset
        X = pd.Series(["mangan", "turu", "adus", "ngomong", "sinau"])
    
    # Initialize vectorizer and model
    tfidf_vectorizer = model.TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(X)
    kmeans = model.KMeans(n_clusters=7, random_state=42)
    kmeans.fit(tfidf_matrix)
    
    # Generate accuracy metrics
    accuracy_metrics = calculate_accuracy()

def calculate_accuracy():
    """Calculate accuracy metrics for display on the accuracy page"""
    # Generate synthetic test data
    words = X.iloc[101:700].values
    words = words.tolist()
    words_low = [word.lower() for word in words]
    synthetic_data = [(model.generate_typo(word), word) for word in words_low]
    
    test_words = [typo for typo, _ in synthetic_data]
    true_corrections = [true for _, true in synthetic_data]
    
    # Evaluate the model
    confusion_matrix = model.evaluate_model_with_confusion_matrix(
        model.kmeans_correction, X, test_words, true_corrections, 
        tfidf_vectorizer, kmeans
    )
    
    # Calculate metrics
    tp, fn, fp, tn = confusion_matrix
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "confusion_matrix": confusion_matrix,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1_score * 100, 2)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html', metrics=accuracy_metrics)

@app.route('/correct_text', methods=['POST'])
def correct_text():
    if request.method == 'POST':
        input_text = request.form['input_text']
        corrected_text = model.kmeans_correction(input_text, X, kmeans, tfidf_vectorizer)
        return jsonify({
            'original': input_text,
            'corrected': corrected_text,
            'changed': input_text != corrected_text
        })