from joblib import load

# Load the saved model
model_filename = 'nithu.joblib'  # Replace with actual path
model = load(model_filename)

# Load the saved vectorizer
vectorizer_filename = 'papu.joblib'  # Replace with actual path
vectorizer = load(vectorizer_filename)

# Function to predict dark patterns
def predict_dark_patterns(texts):
    # Transform the texts to the TF-IDF format using the loaded vectorizer
    texts_tfidf = vectorizer.transform(texts)
    
    # Predict using the loaded model
    predictions = model.predict(texts_tfidf)
    
    return predictions

# Example usage
example_texts = [
    "Chithu🔞 weds nithiya🔪👀and swathi 🥹"
    "Hurry up! This offer ends soon!",
    "Enjoy free shipping on all orders over $50.",
    "Best product at the lowest price!"
]
predictions = predict_dark_patterns(example_texts)

# Displaying the predictions
for text, prediction in zip(example_texts, predictions):
    print(f"Text: {text}\nPrediction: {'Dark Pattern' if prediction == 1 else 'Not a Dark Pattern'}\n")
