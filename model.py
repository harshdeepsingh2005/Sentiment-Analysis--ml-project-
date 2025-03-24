from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
tf.random.set_seed(42)  
  
file_path = "training.1600000.processed.noemoticon.csv" 

df = pd.read_csv(file_path, encoding="latin1", header=None)
df.columns = ["target", "id", "date", "flag", "user", "text"]

data_sample = df.sample(n=5000, random_state=42)[["target", "text"]]
data_sample["target"] = data_sample["target"].apply(lambda x: 1 if x == 4 else 0)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data_sample["text"])
sequences = tokenizer.texts_to_sequences(data_sample["text"])

max_length = 50
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data_sample["target"].values, test_size=0.2, random_state=42)

vocab_size = 5000
embedding_dim = 16

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Flatten(),  
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test), verbose=1)

new_sentences = input("Enter a sentence to analyze sentiment: ")
new_sequences = tokenizer.texts_to_sequences([new_sentences])
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding='post')
predictions = model.predict(new_padded)

def interpret_sentiment(score):
    if score < 0.5:
        return "Highly Sad"
    else:
        return "Highly Happy"

print("Predictions:")
for prediction in predictions:
    sentiment = interpret_sentiment(prediction[0])
    print()
    print()
    print(new_sentences)
    print(f"Sentiment Score: {prediction[0]:.2f} -> {sentiment}")
