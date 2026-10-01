import numpy as np
import matplotlib.pyplot as plt

# Create text to vector dataset

data = {
    # Animals
    "cat": [2, 3],
    "dog": [3, 4],
    "lion": [4, 5],
    "tiger": [4, 4],

    # Vehicles
    "car": [9, 1],
    "bus": [8, 2],
    "bike": [7, 2],
    "train": [9, 3],

    # Food
    "pizza": [2, 8],
    "burger": [3, 8],
    "rice": [2, 7],
    "bread": [3, 7],

    # Technology
    "computer": [8, 8],
    "laptop": [7, 8],
    "phone": [8, 7],
    "robot": [9, 8],

    # Sports
    "football": [5, 2],
    "cricket": [5, 3],
    "tennis": [6, 2],
    "basketball": [6, 3]
}

# Display each word and vector

print("TEXT TO VECTOR EXPLORER")
print("-----------------------")

for word, vector in data.items():
    print(word, "->", vector)

# Display vector dimensions

print("\nVECTOR DIMENSIONS")
print("-----------------")

for word, vector in data.items():
    print(word, "Dimension:", len(vector))

# Compare Two vectors
word1 = input("\nEnter first word: ").lower()
word2 = input("Enter second word: ").lower()

if word1 in data and word2 in data:

    vector1 = np.array(data[word1])
    vector2 = np.array(data[word2])

    print("\nFirst Vector:", vector1)
    print("Second Vector:", vector2)

    # Calculate Euclidean distance

    distance = np.linalg.norm(vector1 - vector2)

    print("Distance between", word1, "and", word2, "=", distance)

else:
    print("Word not found in dataset.")

# Find closest vector

print("\nCLOSEST VECTOR")
print("--------------")

for word in data:

    current_vector = np.array(data[word])

    closest_word = None
    minimum_distance = float("inf")

    for other_word in data:

        if word != other_word:

            other_vector = np.array(data[other_word])

            distance = np.linalg.norm(
                current_vector - other_vector
            )

            if distance < minimum_distance:
                minimum_distance = distance
                closest_word = other_word

    print(
        word,
        "is closest to",
        closest_word,
        "- Distance:",
        round(minimum_distance, 2)
    )

# 2D Visualization of Vectors

plt.figure(figsize=(10, 8))

for word, vector in data.items():

    x = vector[0]
    y = vector[1]

    plt.scatter(x, y)

    plt.text(
        x + 0.1,
        y + 0.1,
        word
    )

plt.title("Text-to-Vector Explorer")
plt.xlabel("Vector Dimension 1")
plt.ylabel("Vector Dimension 2")

plt.grid()

plt.show()



