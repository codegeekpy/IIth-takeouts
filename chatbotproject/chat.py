import nltk #tokanizer library
import string
from sklearn.feature_extraction.text import TfidfVectorizer #try to identify each word repeted in file like and if to, and rank unique words
from nltk.stem import WordNetLemmatizer #convert words into short form like if you have best it will give good for past and future conversion
from nltk.corpus import stopwords #where exactly to stop for chatbot to end the conversation
import numpy as np #for converting text to numerical data and comopare

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# Preprocessing utilities
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    """Preprocess text by removing punctuation, lemmatizing, and removing stop words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load chat pairs and preprocess them
def load_chat_pairs(file_path):
    """
    Load pairs of patterns and responses from a text file.
    Each line should have the pattern and response separated by a '|'.
    """
    patterns = []
    responses = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and '|' in line:
                pattern, response = line.split('|', 1)
                patterns.append(preprocess(pattern.strip()))
                responses.append(response.strip())
    return patterns, responses

# Create a chatbot using vectorization
def start_chat_with_vectorization(file_path):
    """
    Create a chatbot using vectorization and lemmatization principles.
    creating the text to numerical data here
    """
    print("Loading chat pairs from file...")
    patterns, responses = load_chat_pairs(file_path)

    # Vectorize patterns
    vectorizer = TfidfVectorizer() #
    vectors = vectorizer.fit_transform(patterns)

    print("Chatbot: Hi! I am ready to chat. Type 'bye' to exit.")
    while True:
        user_input = input("You: ").lower().strip()
        if user_input in ["bye", "exit", "quit"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        else:
            # Preprocess user input
            user_vector = vectorizer.transform([preprocess(user_input)]) #checking the similarity taking and comparing

            # Compute cosine similarity
            cosine_similarities = np.dot(user_vector, vectors.T).toarray()[0]#similarity checked here

            # Find the best match
            best_match_index = np.argmax(cosine_similarities)
            best_match_score = cosine_similarities[best_match_index]

            if best_match_score > 0.1:  # Threshold for response
                print(f"Chatbot: {responses[best_match_index]}")
            else:
                print("Chatbot: I don't understand that. Could you rephrase?")

if __name__ == "__main__":
    # Replace 'chat_pairs.txt' with the path to your text file
    file_path = r"C:\Users\HP\Downloads\chat_pairs.txt"
    start_chat_with_vectorization(file_path)
