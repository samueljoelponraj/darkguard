import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

# Load the CSV file
file_path = 'dark.csv'  # Replace this with the actual path to your CSV file
df = pd.read_csv(file_path)

# Data Preprocessing and Feature Extraction
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text_content'], df['is_dark_pattern'], test_size=0.2, random_state=42)

# Using TF-IDF to convert text data to numeric format
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Displaying the results
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Saving the trained model and vectorizer to files
model_file_path = 'nithiya1.joblib'  # Replace with desired file path
vectorizer_file_path = 'papi1.joblib'  # Replace with desired file path

dump(model, model_file_path)
dump(vectorizer, vectorizer_file_path)
