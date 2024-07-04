from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label']
