# Bayyin
An AI powered Tool for Readability Assessment and Text Simplification for the Arabic Language


**Bayyin (بيّن)** is an AI-powered tool designed to assess the readability of Arabic texts and simplify them for various proficiency levels. [cite_start]This project addresses the critical gap in Arabic NLP by providing accurate, context-aware evaluations to make Arabic content more accessible to learners, children, and individuals with limited literacy[cite: 24, 27].

[cite_start]Our motivation stems from a desire to reconnect Arab youth with their language by making complex texts more approachable without losing their richness[cite: 46, 51].

---

## 🎯 Project Aim & Objectives

[cite_start]The main goal is to develop an AI tool that performs both **readability assessment** and **text simplification** for the Arabic language[cite: 45].

Our key objectives include:
* [cite_start]Designing a classifier to categorize Arabic texts into clear proficiency levels[cite: 53].
* [cite_start]Comparing various ML, Deep Learning, and Transformer models to find the most effective one[cite: 54].
* [cite_start]Evaluating multiple pre-processing and text simplification strategies[cite: 55, 56].

---

## 🧐 Problem Definition

[cite_start]Assessing Arabic text readability is challenging due to the language's unique characteristics like rich morphology and syntactic variation[cite: 42]. [cite_start]Current tools are often too simplistic, creating a mismatch between text difficulty and reader ability, which can hinder learning and engagement[cite: 23, 28, 40]. Bayyin aims to solve this by providing a nuanced, AI-driven solution.

---

## ✨ Features

* [cite_start]**Multi-Level Readability Classification**: Classifies Arabic text into 7 distinct readability levels using the BAREC dataset[cite: 218].
* **Text Simplification**: Implements multiple strategies to simplify complex texts:
    * **Lexical Simplification**: Replaces difficult words with simpler synonyms.
    * [cite_start]**Syntactic Simplification**: Uses techniques like "Split-and-Rephrase" to restructure complex sentences[cite: 240].
* **State-of-the-Art Models**: Leverages a range of models from classical Machine Learning to advanced Transformers for high accuracy.

---

## 🛠️ Methodology

Our methodology is a multi-step process involving data collection, preprocessing, model training, and simplification.

#### 1. Dataset
[cite_start]We are using the **BAREC dataset**, which contains 69,441 manually annotated Arabic sentences across seven readability levels[cite: 218, 219]. [cite_start]We also plan to expand this dataset to improve model performance[cite: 220].

#### 2. Pre-processing
[cite_start]To ensure clean data, our pipeline includes[cite: 223, 224]:
* Normalization (removing diacritics, punctuation, etc.).
* Tokenization and Segmentation.
* Lemmatization and Morphological Analysis.
* Named Entity Recognition (NER).

#### 3. Models
We will experiment with and compare the following models:

* [cite_start]**Machine Learning**: Support Vector Machine (SVM), Random Forest (RF), and XGBoost trained on linguistic features[cite: 237].
* [cite_start]**Deep Learning**: Graph Neural Networks (GNN), TextCNN, and Bi-LSTM to capture structural and sequential patterns[cite: 238].
* [cite_start]**Transformers**: Fine-tuning state-of-the-art Arabic models like AraBERTv2, CAMELBERT-MSA, and CAMELBERTmix for deep contextual understanding[cite: 239].

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* pip & virtualenv

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/bayyin-arabic-readability.git](https://github.com/your-username/bayyin-arabic-readability.git)
    cd bayyin-arabic-readability
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To classify a sentence, run the `classify.py` script:
```bash
python src/classify.py --text "النص العربي المراد تحليله"
```
To simplify a sentence, run the `simplify.py` script:
```bash
python src/simplify.py --text "النص العربي المعقد المراد تبسيطه"
```

---

## 👥 Project Team

[cite_start]This project is submitted for the fulfillment of the requirements for the graduation project at the University of Jeddah[cite: 2, 6, 7].

**Team Members:**
* [cite_start]Sarah F. Alhalees (2219288) [cite: 5]
* [cite_start]Nagham A. Alshbrawi (2219273) [cite: 5]
* [cite_start]Raya Y. Abu Aljamal (2310903) [cite: 5]
* [cite_start]Fatimah M. Alsinan (2310303) [cite: 5]
* [cite_start]Feryal E. Jadallah (2311180) [cite: 5]
* [cite_start]Bayan Z. Barmeem (2219206) [cite: 5]

**Supervisor:**
* [cite_start]Dr. Shahd Alahdal [cite: 8]

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.