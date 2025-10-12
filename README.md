# Bayyin: An AI-Powered Tool for Arabic Readability Assessment and Text Simplification

**Bayyin (بيّن)** is an AI-powered tool designed to assess the readability of Arabic texts and simplify them for various proficiency levels. This project addresses the critical gap in Arabic NLP by providing accurate, context-aware evaluations to make Arabic content more accessible to learners, children, and individuals with limited literacy.

Our motivation stems from a desire to reconnect Arab youth with their language by making complex texts more approachable without losing their richness.

---

## Project Aim & Objectives

The main goal is to develop an AI tool that performs both **readability assessment** and **text simplification** for the Arabic language.

Our key objectives include:
* Designing a classifier to categorize Arabic texts into clear proficiency levels.
* Comparing various ML, Deep Learning, and Transformer models to find the most effective one.
* Evaluating multiple pre-processing and text simplification strategies.

---

## Problem Definition

Assessing Arabic text readability is challenging due to the language's unique characteristics like rich morphology and syntactic variation. Current tools are often too simplistic, creating a mismatch between text difficulty and reader ability, which can hinder learning and engagement. Bayyin aims to solve this by providing a nuanced, AI-driven solution.

---

## Features

* **Multi-Level Readability Classification**: Classifies Arabic text into 7 distinct readability levels using the BAREC dataset.
* **Text Simplification**: Implements multiple strategies to simplify complex texts:
    * **Lexical Simplification**: Replaces difficult words with simpler synonyms.
    * **Syntactic Simplification**: Uses techniques like "Split-and-Rephrase" to restructure complex sentences.
* **State-of-the-Art Models**: Leverages a range of models from classical Machine Learning to advanced Transformers for high accuracy.

---

## Methodology

Our methodology is a multi-step process involving data collection, preprocessing, model training, and simplification.

#### 1. Dataset
We are using the **BAREC dataset**, which contains 69,441 manually annotated Arabic sentences across seven readability levels. We also plan to expand this dataset to improve model performance.

#### 2. Pre-processing
To ensure clean data, our pipeline includes:
* Normalization (removing diacritics, punctuation, etc.).
* Tokenization and Segmentation.
* Lemmatization and Morphological Analysis.
* Named Entity Recognition (NER).

#### 3. Models
We will experiment with and compare the following models:

* **Machine Learning**: Support Vector Machine (SVM), Random Forest (RF), and XGBoost trained on linguistic features.
* **Deep Learning**: Graph Neural Networks (GNN), TextCNN, and Bi-LSTM to capture structural and sequential patterns.
* **Transformers**: Fine-tuning state-of-the-art Arabic models like AraBERTv2, CAMELBERT-MSA, and CAMELBERTmix for deep contextual understanding.

---

## Getting Started

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



--

## Project Team

This project is submitted for the fulfillment of the requirements for the graduation project at the University of Jeddah.

**Team Members:**
* Sarah F. Alhalees (2219288) 
* Nagham A. Alshbrawi (2219273)
* Raya Y. Abu Aljamal (2310903) 
* Fatimah M. Alsinan (2310303) 
* Feryal E. Jadallah (2311180) 
* Bayan Z. Barmeem (2219206) 

**Supervisor:**
* Dr. Shahd Alahdal 

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
