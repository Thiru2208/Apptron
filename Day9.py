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

# Features:
# [Python, AI, Machine Learning, Sports, Travel]

document_vectors = np.array([
    [1, 0, 0, 0, 0],
    [0.9, 0.1, 0, 0, 0],
    [0, 1, 0.2, 0, 0],
    [0, 0.9, 0.3, 0, 0],
    [0, 0.4, 1, 0, 0],
    [0, 0.3, 0.9, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0.9, 0.1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0.1, 0.9]
])

print("DOCUMENT VECTORS")
print("----------------")

for i in range(len(documents)):
    print(i + 1, ":", documents[i])

print("\nEnter Query Vector")
print("[Python, AI, Machine Learning, Sports, Travel]")

query_vector = []

for i in range(5):
    while True:
        try:
            value = float(input(f"Enter value {i + 1}: "))
            query_vector.append(value)
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value only.")

query_vector = np.array(query_vector)

print("\nQuery Vector:")
print(query_vector)

# Check whether query vector is all zeros
if np.linalg.norm(query_vector) == 0:
    print("\nQuery vector cannot contain only zeros.")

else:
    similarities = []

    for vector in document_vectors:
        dot_product = np.dot(query_vector, vector)

        query_magnitude = np.linalg.norm(query_vector)
        document_magnitude = np.linalg.norm(vector)

        cosine_similarity = dot_product / (
            query_magnitude * document_magnitude
        )

        similarities.append(cosine_similarity)

    similarities = np.array(similarities)

    print("\nCOSINE SIMILARITY SCORES")
    print("------------------------")

    for i in range(len(documents)):
        print(
            "Document",
            i + 1,
            ":",
            round(similarities[i], 4)
        )

    # Sort documents from highest similarity to lowest
    ranked_indices = np.argsort(similarities)[::-1]

    print("\nTOP 3 MOST SIMILAR DOCUMENTS")
    print("----------------------------")

    for rank in range(3):
        index = ranked_indices[rank]

        print(
            f"{rank + 1}. {documents[index]}"
        )

        print(
            "Similarity:",
            round(similarities[index], 4)
        )