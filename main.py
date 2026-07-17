import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ==================== STEP 1: LOAD & CLEAN DATA ====================
# 1. Load the Amazon Alexa Reviews dataset (tab-separated file)
df = pd.read_csv("amazon_alexa.tsv", sep='\t')   # columns: rating, date, variation, verified_reviews, feedback

# 2. Remove empty reviews (agar koi row mein review missing ho)
df = df.dropna(subset=["verified_reviews"])

# 3. Remove neutral reviews (rating == 3), keep only positive/negative
df = df[df["rating"] != 3]

df["sentiment"] = df["rating"].apply(
    lambda x: "positive" if x >= 4 else "negative"
)

# 4. Text cleaning function to remove HTML tags and special characters
def clean_text(text):
    text = str(text).lower()
    
    # Handle contractions (VERY IMPORTANT)
    text = text.replace("won't", "will not")
    text = text.replace("can't", "can not")
    text = text.replace("n't", " not")
    
    # Remove HTML
    text = re.sub(r'<.*?>', '', text)
    
    # Keep words only
    text = re.sub(r'[^a-z\s]', '', text)
    
    return text

# 5. Apply the cleaning function to the 'verified_reviews' column
df["clean_review"] = df["verified_reviews"].apply(clean_text)
print("Step 1 Complete: Data successfully cleaned!")
print(df["sentiment"].value_counts())   # dekho kitni positive/negative/neutral hain


# ==================== STEP 2: CONVERT TEXT TO NUMBERS (TF-IDF) ====================
# 1. Initialize TF-IDF Vectorizer (limiting to max 5000 unique words)
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),   # bigrams
    stop_words=None      # "not" ko remove mat karo
)

# 2. Convert the cleaned text into numbers (matrix)
X = tfidf.fit_transform(df["clean_review"])
y = df["sentiment"]

# 3. Split data into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Print shapes to verify the process
print("Step 2 Complete: Text successfully converted to numbers!")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==================== STEP 3: TRAIN & COMPARE TWO MODELS ====================
print("\nStarting Model Training... Please wait a moment...")

# 1. Model 1: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

# 2. Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# 3. Print detailed classification metrics for Logistic Regression
print("\nDetailed Classification Report (Logistic Regression):")
print(classification_report(y_test, lr_pred))

import pickle
from sklearn.model_selection import cross_val_score

# ==================== STEP 4: CROSS-VALIDATION ====================
print("\nStarting Cross-Validation (Logistic Regression)... Please wait...")
scores = cross_val_score(lr_model, X, y, cv=5)
print("Average CV Accuracy:", scores.mean())


# ==================== STEP 5: SAVE THE MODEL & VECTORIZER ====================
print("\nSaving the model and vectorizer files...")
pickle.dump(lr_model, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))
print("Step 4 & 5 Complete: Model and Vectorizer saved successfully!")