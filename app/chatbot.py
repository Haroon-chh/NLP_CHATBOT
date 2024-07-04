import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv('Q&A.csv')

def get_response(query):
    vectorizer = CountVectorizer().fit_transform(data['question'])
    vectors = vectorizer.transform([query])
    similarities = cosine_similarity(vectors, vectorizer.transform(data['question'])).flatten()
    max_similarity_index = similarities.argmax()
    return data['answer'][max_similarity_index]
