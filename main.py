import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved model, vectorizer, and selected features
with open("sentiment_model.pkl", "rb") as model_file:
    final_model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
with open("selected_features.pkl", "rb") as features_file:
    selected_features = pickle.load(features_file)

# Define selected emotions
selected_emotions = ['neutral', 'grief', 'anger', 'fear', 'surprise', 'disgust', 'sadness']

# Function to analyze user input
def analyze_user_input():
    user_input = input("Enter a sentence to analyze its emotion: ")
    user_input_vect = vectorizer.transform([user_input]).toarray()
    user_input_selected = user_input_vect[:, selected_features]
    prediction = final_model.predict(user_input_selected)
    prediction_proba = final_model.predict_proba(user_input_selected)  # Get probabilities

    # Output probabilities for selected emotions
    emotion_scores = {selected_emotions[i]: prediction_proba[0][i] for i in range(len(selected_emotions))}
    print("Emotion scores (probabilities):")
    for emotion, score in emotion_scores.items():
        print(f"{emotion}: {score:.4f}")

    # Determine the main emotion
    main_emotion = max(emotion_scores, key=emotion_scores.get)
    print(f"The main emotion is: {main_emotion}")

# Analyze user input
analyze_user_input()
