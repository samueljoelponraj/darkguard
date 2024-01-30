import requests
from bs4 import BeautifulSoup
from joblib import load

# Function to scrape text from a website
def scrape_website_text(url):
    # Send a request to the website
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load website: {url}")

    # Parse the website content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    texts = [element.get_text().strip() for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'li'])]  # Adjust tags as needed
    return texts

# Function to load model and vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path):
    model = load(model_path)
    vectorizer = load(vectorizer_path)
    return model, vectorizer

# Function to predict dark patterns
def predict_dark_patterns(texts, model, vectorizer):
    texts_tfidf = vectorizer.transform(texts)
    predictions = model.predict(texts_tfidf)
    return predictions

# Main function to detect dark patterns in a website
def detect_dark_patterns_in_website(url, model_path, vectorizer_path):
    texts = scrape_website_text(url)
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    predictions = predict_dark_patterns(texts, model, vectorizer)

    dark_pattern_texts = [text for text, prediction in zip(texts, predictions) if prediction == 0.9]
    return dark_pattern_texts

# Example usage
website_url = 'https://www.tokopedia.com/'  # Replace with the actual website URL
model_path = '1.joblib'  # Replace with actual model path
vectorizer_path = '2 .joblib'  # Replace with actual vectorizer path

dark_patterns = detect_dark_patterns_in_website(website_url, model_path, vectorizer_path)
print("Detected dark patterns:\n", "\n".join(dark_patterns))
