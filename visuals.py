# Plot Precision, Recall, and F1-Score for All Models
import matplotlib.pyplot as plt
import seaborn as sns

metrics = {
    'Model': ['NaijaBERT', 'VADER', 'Logistic Regression'],
    'Accuracy': [0.9918, 0.4779, 0.9500],
    'Precision': [0.99, 0.46, 0.95],
    'Recall': [0.99, 0.45, 0.94],
    'F1-Score': [0.99, 0.45, 0.94]
}

df_metrics = pd.DataFrame(metrics)

# Bar plot
sns.barplot(data=df_metrics.melt('Model', var_name='Metric', value_name='Score'), 
            x='Metric', y='Score', hue='Model')
plt.title("Comparison of Model Performance")
plt.savefig('/content/drive/MyDrive/JeremyDissertation/metrics_comparison.png')
plt.show()
