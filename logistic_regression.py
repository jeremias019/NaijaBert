# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['Text']).toarray()
X_test = vectorizer.transform(test_data['Text']).toarray()

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, train_data['Label'])
lr_predictions = lr_model.predict(X_test)

# Save Logistic Regression Results
test_data['Logistic_Predicted_Label'] = lr_predictions
