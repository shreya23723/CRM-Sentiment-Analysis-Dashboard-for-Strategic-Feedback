import nltk
nltk.download('stopwords')
nltk.download('punkt')
import pandas as pd
from transformers import pipeline, AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Load the dataset
df = pd.read_csv('C:/Users/Aryan/Desktop/Sentiment Analysis/dataset/amazon_review.csv')

# Select only the review column (replace 'reviewText' with your actual column name)
reviews = df['reviewText'].astype(str)  # Ensure all values are strings

# Optional: Remove any NaN or missing values
reviews = reviews.dropna()

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Initialize the tokenizer for truncation
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Function to remove stop words from the text
stop_words = set(stopwords.words('english'))

def without_stopping_words(text):
    text = word_tokenize(text)
    return " ".join([word for word in text if word.lower() not in stop_words])

# Function to split text into sentences
def split_sentences(sentence):
    result = re.split(r"[.|!|?]", sentence)
    return result

# Function to manually truncate text to ensure it's within max_length
def manual_truncate(text, max_length=512):
    encoding = tokenizer(text, truncation=False, padding=False, max_length=max_length, return_tensors="pt")
    
    # If tokenized length exceeds the max_length, truncate
    while len(encoding['input_ids'][0]) > max_length:
        text = tokenizer.decode(encoding['input_ids'][0][:max_length-1], skip_special_tokens=True)  # Re-truncate
        encoding = tokenizer(text, truncation=False, padding=False, max_length=max_length, return_tensors="pt")
    
    return text

# Function to process reviews in batches with truncation and sentiment analysis
def process_reviews_in_batches(reviews, batch_size=32, max_length=512):
    results = []
    
    for i in range(0, len(reviews), batch_size):
        batch_reviews = reviews[i:i + batch_size].tolist()  # Convert batch to list of strings
        
        # Remove stopwords from each review
        batch_reviews_no_stopwords = [without_stopping_words(review) for review in batch_reviews]
        
        # Manually truncate each review if it exceeds max_length
        batch_reviews_truncated = [manual_truncate(review, max_length=max_length) for review in batch_reviews_no_stopwords]
        
        # Tokenize the batch with truncation and padding
        encodings = tokenizer(batch_reviews_truncated, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        
        # Run sentiment analysis on the batch
        batch_results = sentiment_pipeline(batch_reviews_truncated)
        results.extend(batch_results)
    
    return results

# Process the reviews in batches
results = process_reviews_in_batches(reviews, batch_size=32)

# Add sentiment analysis results to the dataframe
df['sentiment'] = [result['label'] for result in results]

# Save the results to a new CSV file
df.to_csv('sentiment_analysis_results.csv', index=False)
