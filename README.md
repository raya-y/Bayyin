# Bayyin: Arabic Readability Assessment & Text Simplification

**Bayyin (بيّن)** is an AI-powered tool designed to assess the readability of Arabic texts and simplify them for various proficiency levels. This project addresses the critical gap in Arabic NLP by providing accurate, context-aware evaluations to make Arabic content more accessible to learners, children, and individuals with limited literacy.

Our motivation stems from a desire to reconnect Arab youth with their language by making complex texts more approachable without losing their richness.


**Important:** This repository is primarily an archive/collection of notebooks, data artifacts, and experiment outputs. The files are integrated into a runnable application using the trained models on an application; notebooks and scripts are snapshots of experiments. To run experiments, you'll typically need to combine notebooks, adapt paths, and create an execution environment (see the Notes section below).

**Status:** Active research / academic project.

## Project Aim & Objectives

The main goal is to develop an AI tool that performs both **readability assessment** and **text simplification** for the Arabic language.

Our key objectives include:
* Designing a classifier to categorize Arabic texts into clear proficiency levels.
* Comparing various ML, Deep Learning, and Transformer models to find the most effective one.
* Evaluating multiple pre-processing and text simplification strategies.


## Problem Definition

Assessing Arabic text readability is challenging due to the language's unique characteristics like rich morphology and syntactic variation. Current tools are often too simplistic, creating a mismatch between text difficulty and reader ability, which can hinder learning and engagement. Bayyin aims to solve this by providing a nuanced, AI-driven solution.
---
## Quick start

```bash
git clone https://github.com/raya-y/Bayyin.git
cd Bayyin
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dependencies

See `requirements.txt` (pandas, numpy, scikit-learn, xgboost, nltk, torch, transformers, tensorflow, datasets, jupyterlab, matplotlib, seaborn).
---
## Repository structure & file descriptions

- **LICENSE**: MIT license for the project.
- **requirements.txt**: Python packages required to run notebooks and experiments.
- **README.md**: This file — rewritten to reflect the current structure and brief descriptions.

- **Dataset/**: Contains dataset files and notebooks used to prepare and inspect data.
  - `Arabic E-Book Corpus/` : (folder) raw or exported e-book data used for experiments (level 6).
  - `BAREC&DARES.ipynb` : Notebook for exploring or combining the BAREC and DARES datasets.
  - `TheDataset.ipynb`, `TheDataset-5.ipynb` : Notebooks used to inspect, clean, or generate features from datasets.
  - `X_resampled_features.npz` : Numpy compressed file containing resampled feature matrices used for model training.
  - `y_resampled_labels.csv` : CSV file containing corresponding labels after resampling.

- **ML_Models/**: Classical machine-learning model experiments.
  - **Random_Forest/**
    - `RandomForest.ipynb`, `RandomForestFinal.ipynb`, `ml-rf (1).ipynb` : Random Forest feature experiments and final evaluation notebooks.
  - **SVM/**
    - `SVM_Readability_Classifier-2.ipynb` : SVM experiments for classification.
  - **XGBoost/**
    - `XGBoost_Readability_Classifier (2).ipynb` : XGBoost model experiments.

- **DL_Models/**: Deep learning experiments and notebooks.
  - **BiLSTM/**
    - `dl-bilstm (3).ipynb` : BiLSTM model experiments for readability classification.
  - **GNN/**
    - `gnn-bayyin (2).ipynb` : Graph Neural Network experiments.
  - **TextCNN/**
    - `bayyin-textcnn (2).ipynb` : TextCNN model experiments.
    - `Code/` : Supporting code or utilities for the TextCNN experiments.

- **Transformer_Models/**: Transformer fine-tuning experiments using Arabic pretrained models.
  - **AraBERTv2/**
    - `arabertv2 (4).ipynb` : Fine-tuning/evaluation with AraBERTv2.
  - **CAMeLBERT-MSA/**
    - `CAMeLBERT_msa_Bayyin (2).ipynb` : Experiments with CAMeLBERT for Modern Standard Arabic.
  - **CAMeLBERTmix/**
    - `CAMeLBERT-mix_Bayyin (1).ipynb` : CAMeLBERT-mix experiments.

## Interface & links

- **Live interface:** https://your-interface.example.com  

- **Interface GitHub repo:** https://github.com/your-org/your-interface-repo  

## Bayyin data

- **Bayyin dataset / data access:** https://your-data-link.example.com  
- **Trained Models on Hugging Face:** https://huggingface.co/Raya-y/Bayyin_models/tree/main
---
## How to reproduce experiments (high level)

1. Prepare dataset: use the file in Bayyin dataset and run it in the notebook that you want to try.
2. Run ML notebooks under `ML_Models/` for baseline models (SVM, Random Forest, XGBoost).
3. Run DL notebooks under `DL_Models/` for deep learning baselines (BiLSTM, TextCNN, GNN).
4. Run transformer fine-tuning in `Transformer_Models/` (requires GPU and larger memory).

## Notes & recommendations

- Notebooks include numbered copies (e.g., `(1)`, `(2)`) — open the most recent or clearly labeled `Final` notebook for results.
- Transformer experiments expect a GPU-enabled environment and may require additional package versions (see `requirements.txt`).
- For reproducible runs, create a Python virtual environment and pin package versions (consider creating `environment.yml` or `pip` constraints).


---

## Contributors & contact
This project is submitted for the fulfillment of the requirements for the graduation project at the University of Jeddah. For questions about reproducing results or data access, open an issue on the repository or contact the repository owner.

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
