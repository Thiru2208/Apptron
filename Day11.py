from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Create Dataset

documents = [

    # Python
    "Python is a popular programming language.",
    "Python is easy to learn and understand.",
    "Python is widely used for data science.",
    "Python supports object-oriented programming.",
    "Python has many useful libraries.",
    "Python is commonly used for automation.",
    "Python can be used for web development.",
    "Python is useful for machine learning.",
    "Python has simple and readable syntax.",
    "Python is used by many developers around the world.",

    # NumPy
    "NumPy is a Python library for numerical computing.",
    "NumPy provides powerful array operations.",
    "NumPy is useful for mathematical calculations.",
    "NumPy arrays are faster than Python lists.",
    "NumPy supports multidimensional arrays.",
    "NumPy is widely used in data science.",
    "NumPy can perform matrix operations.",
    "NumPy provides statistical functions.",
    "NumPy is useful for scientific computing.",
    "NumPy works well with other Python libraries.",

    # Pandas
    "Pandas is a Python library for data analysis.",
    "Pandas provides DataFrame objects.",
    "Pandas is useful for handling tabular data.",
    "Pandas can read CSV files.",
    "Pandas can clean missing data.",
    "Pandas supports data filtering.",
    "Pandas can group and summarize data.",
    "Pandas is useful for data preprocessing.",
    "Pandas can work with Excel files.",
    "Pandas makes data manipulation easier.",

    # Machine Learning
    "Machine learning allows computers to learn from data.",
    "Machine learning is a branch of artificial intelligence.",
    "Supervised learning uses labelled data.",
    "Unsupervised learning uses unlabelled data.",
    "Classification predicts categories.",
    "Regression predicts numerical values.",
    "Machine learning models learn patterns from data.",
    "Random Forest is a machine learning algorithm.",
    "Machine learning is used for prediction.",
    "Machine learning requires training data.",

    # Artificial Intelligence
    "Artificial intelligence allows machines to perform intelligent tasks.",
    "AI is used in image recognition.",
    "AI is used in natural language processing.",
    "AI systems can make automated decisions.",
    "Artificial intelligence is used in robotics.",
    "AI can understand human language.",
    "AI assistants use artificial intelligence.",
    "AI is used in recommendation systems.",
    "Deep learning is a part of artificial intelligence.",
    "AI can solve complex problems."
]


# Create Categories

categories = (
    ["Python"] * 10 +
    ["NumPy"] * 10 +
    ["Pandas"] * 10 +
    ["Machine Learning"] * 10 +
    ["AI"] * 10
)

# Load Embedding Model

model = SentenceTransformer("all-MiniLM-L6-v2")


# Generate Document Embeddings


document_embeddings = model.encode(documents)

# Get User Query

query = input("Enter your search query: ")

# Example:
# What is NumPy used for?


# Generate Query Embedding

query_embedding = model.encode([query])

# Calculate Similarity


similarity_scores = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

# Rank Documents

top_indices = np.argsort(similarity_scores)[::-1][:5]

# Display Top 5 Results

print("\nTOP 5 RELEVANT DOCUMENTS")
print("-------------------------------")

for rank, index in enumerate(top_indices, start=1):

    print("\nRank:", rank)
    print("Document:", documents[index])
    print("Category:", categories[index])
    print("Similarity Score:",
          round(similarity_scores[index], 4))