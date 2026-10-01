from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Create dataset
sentences = [
    "Python is a popular programming language.",
      "Java is used for building many applications.",
    "Programming helps us solve problems using computers.",
    "Functions help organize code into reusable blocks.",
    "Python is commonly used for data analysis.",
    "Developers use Git to manage source code.",

    # Artificial Intelligence
    "Machine learning is a branch of artificial intelligence.",
    "Artificial intelligence allows machines to perform intelligent tasks.",
    "Deep learning uses neural networks.",
    "AI is used in image recognition systems.",
    "Natural language processing helps computers understand human language.",
    "Chatbots use artificial intelligence to communicate with users.",

    # Education
    "Students use online platforms for learning.",
    "Education helps people develop knowledge and skills.",
    "Teachers use technology in modern classrooms.",
    "Online courses allow students to learn from home.",
    "Examinations are used to evaluate student performance.",
    "Universities provide higher education opportunities.",

    # Business
    "Businesses use marketing to attract customers.",
    "Companies use data to make better decisions.",
    "Good customer service improves business success.",
    "Entrepreneurs create new business opportunities.",
    "Digital marketing is important for modern businesses.",
    "Business analytics helps companies understand performance.",

    # Travel
    "Japan is a popular destination for tourists.",
    "Traveling helps people experience different cultures.",
    "Airplanes allow people to travel between countries.",
    "Hotels provide accommodation for travelers.",
    "Tourists often visit famous historical places.",
    "Travel planning helps people organize their trips."
]

# Load embedding models

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate sentence embeddings

sentance_embeddings = model.encode(sentences)
print("Sentence Embeddings Generated Successfully!")

print("\nEmbedding Shape:")
print(sentance_embeddings.shape)

# Store embeddings
stored_embeddings = np.array(sentance_embeddings)

# Get user query
quary = input("\nEnter your query: ")

# Generate query embedding
query_embedding = model.encode([quary])

# Calculate cosine similarity
similarity_scores = cosine_similarity(query_embedding, stored_embeddings)[0]

# Rank results
top_indices = np.argsort(similarity_scores)[::-1]

# Display top 5 results
print("\nTop 5 Semantic Matches")
print("----------------------")

for rank, index in enumerate(top_indices[:5], start=1):
    print(f"Rank {rank}:")
    print("Sentence:", sentences[index])
    print("Similarity Score:", similarity_scores[index])
    print()