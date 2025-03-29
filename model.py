import random
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
import sys

# Load the dataset and take a sample of 20,000 rows
file_path = "combined_goemotions.csv"  # Ensure the CSV file is in the correct directory
df = pd.read_csv(file_path)


# Define emotion columns of interest
all_emotions = list(df.columns[8:])  # Adjust the start index if emotions don't start at column 8
print(f"All emotions in the dataset: {all_emotions}")

# Define the emotions to use (ensure these exist in the dataset)
selected_emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion']

# Check if selected emotions exist in the dataset
if not all(emotion in all_emotions for emotion in selected_emotions):
    missing_emotions = [emotion for emotion in selected_emotions if emotion not in all_emotions]
    raise ValueError(f"The following emotions are missing in the dataset: {missing_emotions}")

# Extract text data and corresponding labels
texts = df['text'].astype(str).tolist()
labels = df[selected_emotions].values  # Use selected emotions as labels

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert text data into numerical features
vectorizer = CountVectorizer(max_features=1000)
X_train_vect = vectorizer.fit_transform(X_train).toarray()
X_test_vect = vectorizer.transform(X_test).toarray()

# Genetic Algorithm Implementation
POPULATION_SIZE = 20
NUM_GENERATIONS = 50
MUTATION_RATE = 0.1

# Initialize population (binary chromosomes representing feature selection)
def initialize_population():
    return [np.random.randint(2, size=X_train_vect.shape[1]) for _ in range(POPULATION_SIZE)]

# Fitness function: evaluates the accuracy of the model with selected features
def fitness_function(chromosome):
    selected_features = np.where(chromosome == 1)[0]
    if len(selected_features) == 0:
        return 0  # Avoid empty feature sets

    X_train_subset = X_train_vect[:, selected_features]
    X_test_subset = X_test_vect[:, selected_features]

    model = OneVsRestClassifier(MultinomialNB())
    model.fit(X_train_subset, y_train)
    y_pred = model.predict(X_test_subset)
    
    return accuracy_score(y_test, y_pred)

# Selection: Tournament selection
def select_parents(population, fitness_scores):
    parents = []
    for _ in range(2):
        candidates = random.sample(list(zip(population, fitness_scores)), k=3)
        parent = max(candidates, key=lambda x: x[1])[0]
        parents.append(parent)
    return parents

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

# Mutation: Flip random bits
def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# Genetic Algorithm execution
def genetic_algorithm():
    population = initialize_population()

    for generation in range(NUM_GENERATIONS):
        fitness_scores = [fitness_function(chromosome) for chromosome in population]

        next_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))

        population = next_population

        best_fitness = max(fitness_scores)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    # Return the best solution
    best_chromosome = population[np.argmax(fitness_scores)]
    return best_chromosome

# Run the Genetic Algorithm
best_solution = genetic_algorithm()

# Evaluate the final model with the best solution
selected_features = np.where(best_solution == 1)[0]
X_train_best = X_train_vect[:, selected_features]
X_test_best = X_test_vect[:, selected_features]

final_model = OneVsRestClassifier(MultinomialNB())
final_model.fit(X_train_best, y_train)

# Save the model, vectorizer, and selected features
with open("sentiment_model.pkl", "wb") as model_file:
    pickle.dump(final_model, model_file)
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
with open("selected_features.pkl", "wb") as features_file:
    pickle.dump(selected_features, features_file)

final_predictions = final_model.predict(X_test_best)
final_accuracy = accuracy_score(y_test, final_predictions)
print(f"Final Accuracy with Selected Features: {final_accuracy}")

# Take user input to check emotion
def analyze_user_input():
    user_input = input("Enter a sentence to analyze its emotion: ")
    user_input_vect = vectorizer.transform([user_input]).toarray()
    user_input_selected = user_input_vect[:, selected_features]
    prediction = final_model.predict(user_input_selected)
    
    # Output scores for selected emotions
    emotion_scores = {selected_emotions[i]: prediction[0][i] for i in range(len(selected_emotions))}
    print("Emotion scores:")
    for emotion, score in emotion_scores.items():
        print(f"{emotion}: {score}")

analyze_user_input()
