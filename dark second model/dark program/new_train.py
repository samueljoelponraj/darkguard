import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = 'dark.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Preprocessing and Feature Extraction
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text_content'], df['dark_pattern_category'], test_size=0.2, random_state=42)

# Using TF-IDF to convert text data to numeric format
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training
model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Displaying the results
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
