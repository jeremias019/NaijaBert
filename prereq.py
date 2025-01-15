# Install necessary libraries
!pip install transformers torch scikit-learn vaderSentiment matplotlib pandas

# Import required modules
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
