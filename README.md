# Resume Dataset Classification

This project aims to classify document images as either resumes or non-resumes using OCR (Optical Character Recognition) and machine learning models. The dataset is organized into two categories: **resume** and **non-resume**. Various machine learning models are trained on the extracted text from these document images to predict the document type.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Dataset Structure](#dataset-structure)
3. [Installation](#installation)
4. [Project Workflow](#project-workflow)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## 1. Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.x
- Jupyter Notebook (optional but recommended)
- Python Libraries:
  - `PIL`
  - `Tesseract`
  - `scikit-learn`
  - `matplotlib`
  - `pandas`
  - `nltk`
  - `shutil`
  
## 2. Dataset Structure

The project is based on two types of datasets: **resume** and **non-resume** document images. The dataset is organized into the following structure:

The script uses **Tesseract OCR** to extract text from the document images and classifies them into the corresponding categories.

## 3. Installation

To set up the environment and dependencies, follow the steps:

1. Install Tesseract OCR:
   ```bash
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr
2. Install Python dependencies:
   ```bash
   pip install pytesseract scikit-learn matplotlib pandas nltk
3. Download necessary NLTK datasets:
   ```bash
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
4. Project Workflow
   Data Preparation:

    Organize the document images into resume and not_resume folders.
    Copy the document images to their respective directories.
   Text Extraction:

    Extract text from each document image using Tesseract OCR.
    Store the extracted text in a DataFrame along with their respective labels.
   Text Preprocessing:

    Convert text to lowercase.
    Remove stop words, punctuations, and digits.
    Tokenize the text and apply lemmatization.
   Data Splitting:

    Split the dataset into training and testing sets (80/20 ratio).
   Feature Extraction:

    Use TF-IDF Vectorization to convert text data into numerical features.      
5. Model Training and Evaluation
- Models Used:

Logistic Regression
Random Forest
- Metrics:

Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
ROC-AUC Curve
- Training:

The models are trained on the TF-IDF vectors of the preprocessed text data.
- Evaluation:

Predictions are made on the test set, and the model performance is evaluated using metrics such as accuracy, confusion matrix, and ROC curve.

6. Results
- Logistic Regression:

Precision, Recall, and F1-Scores for both the resume and not_resume classes are printed along with the ROC-AUC curve.
- Random Forest:

A comparison of the results for Random Forest is provided, with the confusion matrix and ROC-AUC curve displayed.
