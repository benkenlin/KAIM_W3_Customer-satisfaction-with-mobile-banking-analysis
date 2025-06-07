import pandas as pd
import re
import logging
from datetime import datetime
import spacy # Import spacy for tokenization/lemmatization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SpaCy model for tokenization and lemmatization
# Ensure you have it installed: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("SpaCy model 'en_core_web_sm' loaded successfully for preprocessing.")
except Exception as e:
    logging.error(f"Error loading SpaCy model in preprocessing: {e}. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None

def clean_text(text: str) -> str:
    """Basic text cleaning: remove URLs, special characters, extra spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text) # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def tokenize_and_lemmatize(text: str) -> str:
    """Tokenizes, removes stop words, and lemmatizes text using SpaCy."""
    if nlp is None or not isinstance(text, str) or not text.strip():
        return "" # Return empty string for invalid input
    
    doc = nlp(text.lower())
    # Filter out stop words, punctuation, and non-alphabetic tokens.
    # Keep digits, but ensure they are treated as tokens.
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and (token.is_alpha or token.is_digit)]
    return " ".join(tokens)

def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the raw review DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing raw review data.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    if df.empty:
        logging.warning("Input DataFrame for preprocessing is empty.")
        return pd.DataFrame()

    logging.info(f"Starting preprocessing for {len(df)} reviews...")

    # 1. Handle missing data (remove reviews with no content)
    initial_rows = len(df)
    df.dropna(subset=['content'], inplace=True)
    logging.info(f"Removed {initial_rows - len(df)} rows with missing review content. {len(df)} rows remaining.")

    # 2. Remove duplicates
    # A robust way to identify duplicates is based on review content, score, and date, or unique reviewId.
    # Assuming reviewId is unique from the scraper.
    if 'reviewId' in df.columns:
        df.drop_duplicates(subset=['reviewId'], inplace=True)
    else:
        df.drop_duplicates(subset=['content', 'score', 'at', 'bank'], inplace=True)
    logging.info(f"Removed duplicates. {len(df)} rows remaining.")

    # 3. Normalize dates to YYYY-MM-DD format
    # Ensure 'at' column is datetime type first, coercing errors
    df['at'] = pd.to_datetime(df['at'], errors='coerce')
    df.dropna(subset=['at'], inplace=True) # Drop rows where date conversion failed
    df['date'] = df['at'].dt.strftime('%Y-%m-%d')
    logging.info("Dates normalized to YYYY-MM-DD format.")

    # 4. Apply basic text cleaning to the review content
    df['cleaned_review'] = df['content'].apply(clean_text)

    # 5. Apply tokenization and lemmatization for thematic analysis
    df['processed_text'] = df['cleaned_review'].apply(tokenize_and_lemmatize)

    # 6. Select and rename final columns for the cleaned dataset
    # The 'reviewId' column from the scraper is a good candidate for primary key.
    processed_df = df[['reviewId', 'cleaned_review', 'processed_text', 'score', 'date', 'bank', 'source', 'userName', 'appVersion']].copy()
    processed_df.rename(columns={
        'cleaned_review': 'review_text', # This will be the main text for sentiment/display
        'score': 'rating'
    }, inplace=True)
    
    # KPI check: missing data percentage
    missing_text_count = processed_df['review_text'].isnull().sum() + (processed_df['review_text'] == '').sum()
    total_reviews_after_preprocessing = len(processed_df)
    
    if total_reviews_after_preprocessing > 0:
        missing_data_percentage = (missing_text_count / total_reviews_after_preprocessing) * 100
        logging.info(f"Missing/empty review_text after preprocessing: {missing_data_percentage:.2f}%")
        if missing_data_percentage < 5:
            logging.info("KPI met: Less than 5% missing or empty review text after preprocessing.")
        else:
            logging.warning("KPI NOT met: More than 5% missing or empty review text after preprocessing.")
    else:
        logging.warning("No reviews left after preprocessing to check KPI.")

    logging.info(f"Preprocessing complete. Final DataFrame shape: {processed_df.shape}")
    return processed_df

if __name__ == "__main__":
    import os
    print("Running preprocessing module directly (for testing/debugging)...")
    
    os.makedirs('data/processed_reviews', exist_ok=True)

    # Create a dummy DataFrame for testing
    dummy_data = {
        'content': ["Great app!", "Bad service. Check out my website: http://example.com", "Slow login. Bad UI.<br/>", "Great app!", None, "Another review with &amp; special chars."],
        'score': [5, 1, 2, 5, 3, 4],
        'at': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 1), datetime(2023, 1, 5), datetime(2023,1,6)],
        'bank': ['Bank A', 'Bank B', 'Bank A', 'Bank A', 'Bank C', 'Bank B'],
        'reviewId': ['r1','r2','r3','r1','r4','r5'], # r1 is a duplicate content-wise, r4 is None content
        'userName': ['User1', 'User2', 'User3', 'User1', 'User5', 'User6'],
        'appVersion': ['1.0', '1.2', '1.0', '1.0', '1.1', '1.3'],
        'source': ['Google Play Store'] * 6
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    processed_df = preprocess_reviews(dummy_df)
    if not processed_df.empty:
        print("\nProcessed DataFrame head:")
        print(processed_df.head())
        processed_df.to_csv("data/processed_reviews/processed_bank_reviews.csv", index=False)
        print("Processed reviews saved to data/processed_reviews/processed_bank_reviews.csv")
    else:
        print("No processed data to save.")