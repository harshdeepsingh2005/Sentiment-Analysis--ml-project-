import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Load Dataset (Sample 100 rows for faster execution)
data = pd.read_csv('combined_goemotions.csv').sample(10000, random_state=42)

# Define emotion columns
emotion_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 
    'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 
    'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 
    'sadness', 'surprise', 'neutral'
]

# Extract the predominant emotion for each row
data['emotion'] = data[emotion_columns].idxmax(axis=1)

# Example: If you have a text column, keep it. Adjust if the column name differs.
text_column = 'text'  # Replace 'text' with the actual name of the column containing the text data.
data = data[[text_column, 'emotion']]

# 2. Data Preprocessing
# Tokenize and pad text
max_words = 5000  # Reduce vocabulary size for faster processing
max_len = 50      # Reduce sequence length
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(data[text_column])
sequences = tokenizer.texts_to_sequences(data[text_column])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Convert emotions to one-hot encoding
label_mapping = {emotion: idx for idx, emotion in enumerate(emotion_columns)}
data['emotion_encoded'] = data['emotion'].map(label_mapping)
labels = to_categorical(data['emotion_encoded'], num_classes=len(emotion_columns))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# 3. Build the Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(len(emotion_columns), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the Model (Reduced Epochs for Testing)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=16)

# Save the model
model.save('emotion_model_sample.h5')

# 5. Define Prediction for Selected Emotions
def predict_seven_emotions(text):
    # Selected emotions to display
    selected_emotions = ['joy', 'sadness', 'anger', 'love', 'fear', 'surprise']
    
    # Preprocess text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    # Predict emotion scores
    prediction = model.predict(padded_sequence)
    scores = prediction[0]
    
    # Filter scores for the selected emotions
    selected_emotion_scores = {emotion: round(scores[emotion_columns.index(emotion)], 4) 
                                for emotion in selected_emotions}
    
    # Determine the predicted emotion
    predicted_emotion = max(selected_emotion_scores, key=selected_emotion_scores.get)
    
    return predicted_emotion, selected_emotion_scores

# 6. Test Prediction
sample_text = "I am thrilled about this opportunity!"
predicted_emotion, selected_emotion_scores = predict_seven_emotions(sample_text)

print(f"Predicted Emotion: {predicted_emotion}")
print("Emotion Scores:")
for emotion, score in selected_emotion_scores.items():
    print(f"  {emotion}: {score*1000}")
