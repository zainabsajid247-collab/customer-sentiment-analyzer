# Customer Review Sentiment Analyzer

A machine learning web app that predicts whether a customer product review is **positive** or **negative**, built using real Amazon customer review data.

## What It Does
Users type in a product review, and the app instantly predicts the sentiment behind it using a trained NLP model.

## Tools & Technologies
- Python
- Scikit-learn (Logistic Regression, Random Forest)
- TF-IDF Vectorization (text-to-numbers conversion)
- Pandas (data cleaning & processing)
- Streamlit (web app interface)

## Dataset
Amazon Alexa product reviews dataset (real customer reviews with star ratings), sourced from Kaggle.

## Model Performance
- Logistic Regression Accuracy: 92.8%
- Cross-Validation Accuracy: 92.7%
- Negative review detection improved using class-balanced training to handle dataset imbalance (majority of reviews were positive).

## How It Works
1. Raw review text is cleaned (lowercased, contractions expanded, special characters removed)
2. Text is converted into numerical features using TF-IDF (with bigrams)
3. A Logistic Regression model, trained on this data, predicts sentiment
4. Streamlit provides a simple interface for real-time predictions

## How to Run Locally
Install required libraries: pip install streamlit scikit-learn pandas
Then run: streamlit run app.py

## Note on Limitations
The dataset had significantly more positive reviews than negative ones. Class-balancing techniques were applied during training to improve negative sentiment detection. Future improvement: testing with a larger, more balanced dataset.

## Author
Zainab Sajid — BS Artificial Intelligence Student
