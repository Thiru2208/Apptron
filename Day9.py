import numpy as np

# Create 10 short documents 
documents = [
    "Python is a popular programming language.",
    "Python is used for data analysis and automation.",
    "Artificial Intelligence helps machines think.",
    "AI is used in smart systems and robotics.",
    "Machine Learning allows computers to learn from data.",
    "Machine Learning is widely used in prediction systems.",
    "Football is a popular sport around the world.",
    "Cricket is a famous sport in many countries.",
    "Travel helps people explore new places.",
    "Tourism and travel are important for many countries."
]

# Represent each document using numerical vectors
# Features: [Python, AI, Machine Learning, Sport, Travel]

document_vectors = np.array([
    [1, 0, 0, 0, 0],      # Document 1 - Python
    [0.9, 0.1, 0, 0, 0],  # Document 2 - Python
    [0, 1, 0.2, 0, 0],    # Document 3 - AI
    [0, 0.9, 0.3, 0, 0],  # Document 4 - AI
    [0, 0.4, 1, 0, 0],    # Document 5 - Machine Learning
    [0, 0.3, 0.9, 0, 0],  # Document 6 - Machine Learning
    [0, 0, 0, 1, 0],      # Document 7 - Sports
    [0, 0, 0, 0.9, 0.1],  # Document 8 - Sports
    [0, 0, 0, 0, 1],      # Document 9 - Travel
    [0, 0, 0, 0.1, 0.9]   # Document 10 - Travel
])

print("DOCUMENT VECTORS")
print("----------------")

for i in range(len(documents)):
    print(i + 1, ":", documents[i])

# Accept query vector
print("\nEnter Query Vector")
print("[Python, AI, Machine Learning, Sports, Travel]")

query_vector = []

for i in range(5):
    value = float(input(f"Enter value {i + 1}: "))
    query_vector.append(value)

query_vector = np.array(query_vector)

print("\nQuery Vector:")
print(query_vector)

