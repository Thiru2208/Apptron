 
# AI KNOWLEDGE BASE SEARCH SYSTEM
# Complete Embedding + Semantic Search Project
 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
 
# 1. LOAD EMBEDDING MODEL
 
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

 
# 2. CREATE KNOWLEDGE BASE DATASET
 

documents = [

    # PYTHON

    {
        "document_name": "Python Basics",
        "category": "Python",
        "text": "Python is a popular high-level programming language. "
                "It is widely used in artificial intelligence, machine learning, "
                "data science, web development and automation."
    },

    {
        "document_name": "Python Variables",
        "category": "Python",
        "text": "Variables are used to store data in Python. "
                "Python does not require programmers to declare the datatype "
                "of a variable explicitly."
    },

    {
        "document_name": "Python Functions",
        "category": "Python",
        "text": "Functions are reusable blocks of code. "
                "Functions help reduce code duplication and make programs "
                "easier to understand and maintain."
    },

    {
        "document_name": "Python Lists",
        "category": "Python",
        "text": "A Python list is an ordered and mutable collection. "
                "Lists can store multiple values and can contain different datatypes."
    },

    {
        "document_name": "Python Dictionaries",
        "category": "Python",
        "text": "Python dictionaries store information using key value pairs. "
                "Dictionary values can be accessed using their corresponding keys."
    },

    {
        "document_name": "Python Loops",
        "category": "Python",
        "text": "Loops allow programmers to repeatedly execute a block of code. "
                "Python mainly provides for loops and while loops."
    },

    {
        "document_name": "Python Conditions",
        "category": "Python",
        "text": "Conditional statements allow a program to make decisions. "
                "Python uses if, elif and else statements."
    },

    {
        "document_name": "Python Classes",
        "category": "Python",
        "text": "Classes are used in object oriented programming. "
                "A class acts as a blueprint for creating objects."
    },

    {
        "document_name": "Python Exceptions",
        "category": "Python",
        "text": "Exception handling prevents programs from terminating unexpectedly. "
                "Python uses try, except, else and finally blocks."
    },

    {
        "document_name": "Python Files",
        "category": "Python",
        "text": "Python can read and write files using the open function. "
                "Common file modes include read, write and append."
    },


    # ARTIFICIAL INTELLIGENCE / MACHINE LEARNING

    {
        "document_name": "Artificial Intelligence",
        "category": "AI ML",
        "text": "Artificial intelligence enables computers and machines "
                "to perform tasks that normally require human intelligence."
    },

    {
        "document_name": "Machine Learning",
        "category": "AI ML",
        "text": "Machine learning is a branch of artificial intelligence "
                "where computers learn patterns from data without being "
                "explicitly programmed for every situation."
    },

    {
        "document_name": "Supervised Learning",
        "category": "AI ML",
        "text": "Supervised learning trains a machine learning model using "
                "labelled data containing both input features and expected outputs."
    },

    {
        "document_name": "Unsupervised Learning",
        "category": "AI ML",
        "text": "Unsupervised learning works with data that does not contain labels. "
                "It is commonly used for clustering and discovering hidden patterns."
    },

    {
        "document_name": "Classification",
        "category": "AI ML",
        "text": "Classification is a supervised machine learning task used "
                "to predict categories or classes."
    },

    {
        "document_name": "Regression",
        "category": "AI ML",
        "text": "Regression is a supervised machine learning technique used "
                "to predict continuous numerical values."
    },

    {
        "document_name": "Random Forest",
        "category": "AI ML",
        "text": "Random Forest is an ensemble machine learning algorithm. "
                "It combines predictions from multiple decision trees."
    },

    {
        "document_name": "Neural Networks",
        "category": "AI ML",
        "text": "Neural networks are machine learning models inspired by "
                "the structure and operation of the human brain."
    },

    {
        "document_name": "Deep Learning",
        "category": "AI ML",
        "text": "Deep learning uses neural networks with multiple layers "
                "to learn complex representations from large datasets."
    },

    {
        "document_name": "Natural Language Processing",
        "category": "AI ML",
        "text": "Natural language processing helps computers understand, "
                "interpret and generate human language."
    },

    {
        "document_name": "Computer Vision",
        "category": "AI ML",
        "text": "Computer vision allows computers to analyze and understand "
                "images and videos."
    },

    {
        "document_name": "Training Dataset",
        "category": "AI ML",
        "text": "A training dataset is the portion of data used to teach "
                "a machine learning model."
    },

    {
        "document_name": "Testing Dataset",
        "category": "AI ML",
        "text": "A testing dataset is used to evaluate how well a trained "
                "machine learning model performs on unseen data."
    },

    {
        "document_name": "Overfitting",
        "category": "AI ML",
        "text": "Overfitting occurs when a machine learning model learns "
                "the training data too closely and performs poorly on new data."
    },

    {
        "document_name": "Underfitting",
        "category": "AI ML",
        "text": "Underfitting occurs when a model is too simple to capture "
                "important patterns in the training data."
    },

    {
        "document_name": "Model Accuracy",
        "category": "AI ML",
        "text": "Accuracy measures the proportion of predictions that were "
                "correctly classified by a model."
    },

    {
        "document_name": "Feature",
        "category": "AI ML",
        "text": "A feature is an individual measurable property used as "
                "an input to a machine learning model."
    },

    {
        "document_name": "Label",
        "category": "AI ML",
        "text": "A label represents the target value that a supervised "
                "machine learning model attempts to predict."
    },

    {
        "document_name": "Embeddings",
        "category": "AI ML",
        "text": "Embeddings convert text, images or other information into "
                "numerical vectors that capture semantic meaning."
    },

    {
        "document_name": "Semantic Search",
        "category": "AI ML",
        "text": "Semantic search identifies information based on meaning "
                "rather than depending only on exact keyword matching."
    },

    {
        "document_name": "Cosine Similarity",
        "category": "AI ML",
        "text": "Cosine similarity measures the similarity between two vectors "
                "by calculating the cosine of the angle between them."
    },

    {
        "document_name": "Vector Database",
        "category": "AI ML",
        "text": "A vector database stores embeddings and allows applications "
                "to search for vectors that are semantically similar."
    },

    {
        "document_name": "RAG",
        "category": "AI ML",
        "text": "Retrieval Augmented Generation combines information retrieval "
                "with a large language model to generate answers using "
                "relevant retrieved documents."
    },

    # NUMPY

    {
        "document_name": "NumPy Introduction",
        "category": "NumPy",
        "text": "NumPy is a Python library used for numerical computing "
                "and working efficiently with multidimensional arrays."
    },

    {
        "document_name": "NumPy Array",
        "category": "NumPy",
        "text": "A NumPy array stores multiple values in an efficient "
                "multidimensional data structure."
    },

    {
        "document_name": "NumPy Shape",
        "category": "NumPy",
        "text": "The shape attribute returns the dimensions of a NumPy array."
    },

    {
        "document_name": "NumPy Mean",
        "category": "NumPy",
        "text": "The NumPy mean function calculates the arithmetic average "
                "of numerical values."
    },

    {
        "document_name": "NumPy Median",
        "category": "NumPy",
        "text": "The NumPy median function calculates the middle value "
                "of a sorted numerical dataset."
    },

    {
        "document_name": "NumPy Standard Deviation",
        "category": "NumPy",
        "text": "Standard deviation measures how far values are spread "
                "from the average value."
    },

    {
        "document_name": "NumPy Random",
        "category": "NumPy",
        "text": "The NumPy random module is used to generate random values "
                "and random datasets."
    },

    {
        "document_name": "NumPy NaN",
        "category": "NumPy",
        "text": "NaN represents missing or undefined numerical values "
                "in NumPy datasets."
    },

    {
        "document_name": "NumPy Reshape",
        "category": "NumPy",
        "text": "The reshape function changes the dimensions of a NumPy array "
                "without changing its underlying values."
    },

    {
        "document_name": "NumPy Filtering",
        "category": "NumPy",
        "text": "Boolean indexing can be used to filter NumPy arrays "
                "according to specified conditions."
    },

    # PANDAS

    {
        "document_name": "Pandas Introduction",
        "category": "Pandas",
        "text": "Pandas is a Python library used for data manipulation, "
                "data cleaning and data analysis."
    },

    {
        "document_name": "Pandas DataFrame",
        "category": "Pandas",
        "text": "A Pandas DataFrame is a two-dimensional labelled data structure "
                "containing rows and columns."
    },

    {
        "document_name": "Pandas Series",
        "category": "Pandas",
        "text": "A Pandas Series is a one-dimensional labelled data structure."
    },

    {
        "document_name": "Pandas CSV",
        "category": "Pandas",
        "text": "The read_csv function loads CSV files into Pandas DataFrames."
    },

    {
        "document_name": "Pandas Missing Values",
        "category": "Pandas",
        "text": "Pandas provides functions such as isnull, dropna and fillna "
                "for detecting and handling missing data."
    },

    {
        "document_name": "Pandas Filtering",
        "category": "Pandas",
        "text": "Pandas DataFrames can be filtered by applying conditions "
                "to selected columns."
    },

    {
        "document_name": "Pandas GroupBy",
        "category": "Pandas",
        "text": "The groupby function groups data according to categories "
                "and allows aggregation operations."
    },

    {
        "document_name": "Pandas Merge",
        "category": "Pandas",
        "text": "The merge function combines multiple Pandas DataFrames "
                "using one or more common columns."
    },


    # COMPANY POLICIES

    {
        "document_name": "Annual Leave Policy",
        "category": "Company Policy",
        "text": "Employees are entitled to annual leave according to "
                "company policy. Annual leave should normally be requested "
                "and approved before the employee takes leave."
    },

    {
        "document_name": "Sick Leave Policy",
        "category": "Company Policy",
        "text": "Employees who are unable to attend work due to illness "
                "should notify their supervisor as soon as possible."
    },

    {
        "document_name": "Working Hours Policy",
        "category": "Company Policy",
        "text": "Employees are expected to follow the official working hours "
                "specified by the organization."
    },

    {
        "document_name": "Remote Work Policy",
        "category": "Company Policy",
        "text": "Employees may work remotely when permission has been granted "
                "by their manager and the nature of their work allows it."
    },

    {
        "document_name": "Password Policy",
        "category": "Company Policy",
        "text": "Employees should create strong passwords and must not share "
                "their account passwords with other people."
    },

    {
        "document_name": "Data Security Policy",
        "category": "Company Policy",
        "text": "Confidential organizational information must be protected "
                "and should only be accessed by authorized employees."
    },

    {
        "document_name": "Attendance Policy",
        "category": "Company Policy",
        "text": "Employees are expected to attend work regularly and "
                "inform their supervisors if they expect to be absent."
    },

    {
        "document_name": "Code of Conduct",
        "category": "Company Policy",
        "text": "Employees should behave professionally and respectfully "
                "towards colleagues, customers and other stakeholders."
    },

    # TRAINING INFORMATION

    {
        "document_name": "Training Attendance Policy",
        "category": "Training",
        "text": "Employees must attend scheduled training sessions. "
                "If an employee cannot attend a training session, "
                "they should inform the trainer or supervisor in advance "
                "and arrange to attend another available session."
    },

    {
        "document_name": "Training Registration",
        "category": "Training",
        "text": "Employees should register for training sessions before "
                "the registration deadline."
    },

    {
        "document_name": "Training Completion",
        "category": "Training",
        "text": "A training programme is considered completed when "
                "the employee attends the required sessions and completes "
                "all compulsory assessments."
    },

    {
        "document_name": "Training Certificate",
        "category": "Training",
        "text": "Employees who successfully complete an approved training "
                "programme may receive a completion certificate."
    },

    {
        "document_name": "Online Training",
        "category": "Training",
        "text": "Online training allows employees to complete approved "
                "learning activities using the company's online training platform."
    },

    {
        "document_name": "Training Assessment",
        "category": "Training",
        "text": "Some training programmes require employees to complete "
                "a final assessment or practical activity."
    }
]

# 3. CREATE EXTRA DOCUMENTS UNTIL WE HAVE 100+

extra_topics = [
    "Data Science",
    "Data Cleaning",
    "Data Visualization",
    "Matplotlib",
    "Scikit Learn",
    "Decision Tree",
    "K Nearest Neighbors",
    "Support Vector Machine",
    "Clustering",
    "K Means",
    "DBSCAN",
    "Principal Component Analysis",
    "Feature Selection",
    "Feature Engineering",
    "Cross Validation",
    "Precision",
    "Recall",
    "F1 Score",
    "Confusion Matrix",
    "Gradient Descent",
    "Large Language Model",
    "Generative AI",
    "Prompt Engineering",
    "Vector Search",
    "Top K Retrieval",
    "Similarity Threshold",
    "Document Chunking",
    "Metadata",
    "Knowledge Base",
    "Information Retrieval",
    "Database",
    "SQL",
    "MongoDB",
    "API",
    "Flask",
    "FastAPI",
    "Git",
    "GitHub",
    "Model Evaluation",
    "Model Training",
    "Data Preprocessing"
]

for topic in extra_topics:

    documents.append(
        {
            "document_name": topic,
            "category": "General Technology",
            "text": f"{topic} is an important concept used in modern "
                    f"computer science, artificial intelligence, machine "
                    f"learning or software development. Understanding "
                    f"{topic} helps students build practical data and AI systems."
        }
    )

 
# 4. CLEAN TEXT
 
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove unnecessary spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces from beginning and end
    text = text.strip()

    return text

 
# 5. TEXT CHUNKING
 

def split_into_chunks(text, chunk_size=40):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = words[i:i + chunk_size]

        chunk = " ".join(chunk)

        chunks.append(chunk)

    return chunks

 
# 6. PROCESS ALL DOCUMENTS
 

knowledge_base = []

for document in documents:

    cleaned_text = clean_text(document["text"])

    chunks = split_into_chunks(cleaned_text)

    for chunk in chunks:

        knowledge_base.append(
            {
                "document_name": document["document_name"],
                "category": document["category"],
                "chunk": chunk
            }
        )


print("\n--------------------------------------------")
print("KNOWLEDGE BASE INFORMATION")
print("--------------------------------------------")

print("Number of Documents :", len(documents))
print("Number of Chunks    :", len(knowledge_base))


 
# 7. GENERATE DOCUMENT EMBEDDINGS
 

print("\nGenerating document embeddings...")

chunk_texts = [
    item["chunk"]
    for item in knowledge_base
]

document_embeddings = model.encode(chunk_texts)

print("Document embeddings generated successfully!")

print("Embedding shape:", document_embeddings.shape)


 
# 8. SEMANTIC SEARCH FUNCTION
 

def semantic_search(query, top_k=5, similarity_threshold=0.20):

    cleaned_query = clean_text(query)

    # Convert query into embedding
    query_embedding = model.encode([cleaned_query])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    # Sort similarity values from highest to lowest
    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    for index in ranked_indices:

        score = similarities[index]

        if score >= similarity_threshold:

            result = {
                "document_name":
                    knowledge_base[index]["document_name"],

                "category":
                    knowledge_base[index]["category"],

                "chunk":
                    knowledge_base[index]["chunk"],

                "similarity":
                    float(score)
            }

            results.append(result)

        if len(results) == top_k:
            break

    return results


 
# 9. GENERATE SIMPLE AI ANSWER
 

def generate_answer(query, results):

    if len(results) == 0:

        return (
            "I could not find enough relevant information "
            "in the knowledge base."
        )

    best_result = results[0]

    answer = best_result["chunk"]

    return answer


 
# 10. DISPLAY SEARCH RESULTS
 

def display_results(query, results):

    print("\n")
    print("=" * 70)
    print("AI KNOWLEDGE BASE SEARCH")
    print("=" * 70)

    print("\nQuery:")
    print(query)

    if len(results) == 0:

        print("\nNo relevant documents were found.")
        return

    print("\nTOP SEARCH RESULTS")
    print("-" * 70)

    for number, result in enumerate(results, start=1):

        print(f"\nRESULT {number}")

        print(
            "Document Name :",
            result["document_name"]
        )

        print(
            "Category      :",
            result["category"]
        )

        print(
            "Similarity    :",
            round(result["similarity"], 4)
        )

        print(
            "Relevant Text :",
            result["chunk"]
        )

        print("-" * 70)


 
# 11. MAIN PROGRAM
 

def main():

    print("\n")
    print("=" * 70)

    print(
        "        AI KNOWLEDGE BASE SEARCH SYSTEM"
    )

    print("=" * 70)

    print(
        "\nSearch topics such as:"
    )

    print(
        "Python, AI, Machine Learning, NumPy, "
        "Pandas, Company Policies and Training"
    )

    print(
        "\nType 'exit' to stop the program."
    )

    while True:

        print("\n" + "=" * 70)

        query = input(
            "\nEnter your question: "
        )

        if query.lower() == "exit":

            print(
                "\nThank you for using "
                "AI Knowledge Base Search System!"
            )

            break

        if query.strip() == "":

            print(
                "Please enter a valid question."
            )

            continue

        # Search
        results = semantic_search(
            query,
            top_k=5,
            similarity_threshold=0.20
        )

        # Display results
        display_results(
            query,
            results
        )

        # Generate answer
        answer = generate_answer(
            query,
            results
        )

        print("\n")
        print("=" * 70)

        print("AI ANSWER")

        print("=" * 70)

        print(answer)


 
# 12. START PROGRAM
 

if __name__ == "__main__":

    main()