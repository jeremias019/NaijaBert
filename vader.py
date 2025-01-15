# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Function to predict sentiment
def vader_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 2  # Positive
    elif score <= -0.05:
        return 0  # Negative
    else:
        return 1  # Neutral

# Apply VADER to test data
test_data['VADER_Predicted_Label'] = test_data['Text'].apply(vader_sentiment)
