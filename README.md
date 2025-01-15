# Sentiment Analysis Framework: NaijaBERT vs VADER vs Logistic Regression
This repository contains the codebase for a comparative sentiment analysis study utilizing NaijaBERT, VADER, and Logistic Regression models on a dataset comprising tweets discussing Nigeria's 2023 presidential candidates.

The objective of this project is to evaluate the performance of NaijaBERT, a fine-tuned BERT-based sentiment analysis model, against VADER and Logistic Regression, using various metrics such as accuracy, precision, recall, and F1-score. The results include visualization plots, misclassification analysis, and performance summaries to determine the most effective model for sentiment analysis in the context of Nigerian socio-political tweets.

# Repository Structure
The repository is organized into modular Python scripts, each handling a distinct component of the process:

prereq.py: Installs the required libraries and imports dependencies for the project.
dataprep.py: Handles data preprocessing, including cleaning, mapping sentiment labels, and splitting datasets into training and test sets.
naijabert_finetuning.py: Contains code for fine-tuning the NaijaBERT model using the preprocessed dataset.
logistic_regression.py: Implements logistic regression using TF-IDF vectorization for sentiment analysis.
vader.py: Uses the VADER Sentiment Analysis tool to predict sentiment labels.
eval.py: Evaluates the performance of all models, generates classification reports, and computes confusion matrices.
visuals.py: Generates comparative plots for metrics and misclassification rates across the models.

# Features
Data Preprocessing:

Handles cleaning, removing duplicates, and mapping sentiment labels.
Prepares datasets for input into various models.
Model Training:

Fine-tuning NaijaBERT for sentiment analysis.
Training and evaluation of Logistic Regression with TF-IDF.
Sentiment Prediction:

VADER-based sentiment classification.
Evaluation and Visualization:

Generates classification reports, confusion matrices, and performance metrics.
Visualizes results through bar charts and comparative plots.
Usage
Setup:

Run prereq.py to install dependencies:
python prereq.py

Data Preparation:

Execute dataprep.py to preprocess the dataset:
python dataprep.py

Model Training and Evaluation:

Fine-tune NaijaBERT using:

python naijabert_finetuning.py

# Train Logistic Regression:

python logistic_regression.py
Predict sentiment using VADER:

python vader.py

# Evaluation:

Generate evaluation metrics using:
python eval.py

# Visualization:

Produce comparative performance plots:
python visuals.py

# The finetuned model can be found here
https://huggingface.co/jeremias019/naijabert
