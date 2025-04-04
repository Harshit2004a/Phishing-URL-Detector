# Phishing URL Detector.
# Can be trained on new data for better accuracy.
# This Phishing URL Detector uses machine learning to analyze website links and determine whether they are legitimate 🟢 or phishing 🔴.
# Code by Harshit Kumar.

import pandas as pd
import re
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Loading dataset given in the csv file
df = pd.read_csv("phishing_dataset.csv", encoding="utf-8")
df["label"] = df["status"].map({"phishing": 1, "legitimate": 0})
df = df[["url", "label"]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["url"], df["label"], test_size=0.2, random_state=42)

def url_tokenizer(url):
    return re.split(r"[/.?=_&-]+", url)

# Feature extraction with fixes
vectorizer = TfidfVectorizer(
    tokenizer=url_tokenizer,
    token_pattern=None,  # ✅ Fix warning
    ngram_range=(1, 3),
    max_features=7000
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# Training model with class weight :)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weights_dict)
model.fit(X_train_tfidf, y_train)

# Evaluating model accuracy according to the given dataset
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save model & vectorizer
joblib.dump(model, "phishing_detector.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained and saved successfully!")

# Prediction function to state T or F
def predict_url(url):
    url_tfidf = vectorizer.transform([url])
    prediction = model.predict(url_tfidf)[0]
    return "🔴 Phishing" if prediction == 1 else "🟢 Legitimate"

# Ask for user input AFTER displaying accuracy of the model and printing result
user_url = input("Enter a URL to check: ").strip()
print(f"Prediction: {predict_url(user_url)}")
