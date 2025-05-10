# 📖 JAVACURE: Javanese Spell Correction Web Application

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Framework-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

This application is a simple project for performing **spell correction** in the Javanese language using **Levenshtein Distance** and **K-Means Clustering** algorithms. The web-based application is built using **Flask** for the backend and **HTML, CSS, JavaScript** for the frontend.

---

## 📸 Application Preview

![App Screenshot](./static/Screenshot%202025-03-13%20105314.png)
![App Screenshot](./static/Screenshot%202025-03-13%20105105.png)

---

## 🎯 Features

- ✅ Detect and correct words in the Javanese language.
- ✅ Recommend the closest word based on the **Levenshtein Distance** algorithm.
- ✅ Group words using **K-Means Clustering**.
- ✅ Simple and user-friendly web-based interface.

---

## 📦 Dataset

- Uses a **preprocessed dataset of 6046 Javanese words**.
- The original dataset was taken from: [sastra.org](https://www.sastra.org).
- Preprocessing was done using two notebooks:
  - `Proyek Besar fix_00_NLP_buat data.ipynb`
  - `Proyek Besar fix_01_NLP_Preprocessing data.ipynb`

---

## 🔧 Technologies Used

### Backend:

- **Python**
- **Flask** — Python web framework for integrating Python with HTML, CSS, and JavaScript environments.

### Frontend:

- **HTML**
- **CSS**
- **JavaScript**

### Algorithms:

- **Levenshtein Distance** — Calculates the edit distance between words to detect spelling errors.
- **K-Means Clustering** — Groups words based on similarity characteristics.

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/filbertcha/javacure.git
   cd javacure

   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask App:

   ```bash
   python run.py
   ```

4. Open your browser and access `http://localhost:5000`

---

## 📝 Notes
- This website is for educational purposes only and intended solely for learning about Natural Language Processing.

## 📞 Contact

- **Name**: Filbert C. B. Kristianto
- **GitHub**: [https://github.com/filbertcha](https://github.com/filbertcha)

```

```
