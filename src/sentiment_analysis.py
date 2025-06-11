import pandas as pd
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the sentiment analysis pipeline globally to avoid re-loading
# Using 'distilbert-base-uncased-finetuned-sst-2-english' as specified.
# This model outputs 'POSITIVE' or 'NEGATIVE' with a score.
# For 'neutral', we will apply a threshold heuristic.
sentiment_classifier = None
try:
    # Set device to 'cuda' if GPU is available, otherwise 'cpu'
    import torch
    device = 0 if torch.cuda.is_available() else -1
    sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
    logging.info(f"Sentiment analysis model loaded successfully on device: {'CUDA' if device == 0 else 'CPU'}.")
except Exception as e:
    logging.error(f"Error loading sentiment analysis model: {e}")
    logging.warning("Sentiment analysis will use a placeholder if model failed to load.")


def get_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment of a given text using the pre-loaded model.

    Args:
        text (str): The review text.

    Returns:
        dict: A dictionary containing 'label' (POSITIVE/NEGATIVE/NEUTRAL) and 'score'.
    """
    if sentiment_classifier is None or not text or not isinstance(text, str) or not text.strip():
        # Return a neutral default if model failed to load or text is invalid/empty
        return {'label': 'NEUTRAL', 'score': 0.5}

    try:
        # The distilbert-base-uncased-finetuned-sst-2-english model is fine-tuned
        # on a dataset (SST-2) that primarily yields 'POSITIVE' or 'NEGATIVE' labels.
        # To get 'NEUTRAL', we use a heuristic based on the confidence score.
        results = sentiment_classifier(text, top_k=2) # Get top 2 scores to check confidence
        
        # Sort results by score (descending) to ensure we pick the most confident one
        results.sort(key=lambda x: x['score'], reverse=True)
        
        best_result = results[0]
        label = best_result['label']
        score = best_result['score']

        # ADJUSTED HEURISTIC for 'NEUTRAL':
        # Lowering this threshold will classify more reviews as POSITIVE/NEGATIVE
        # If the highest confidence score for POS or NEG is below this, it's considered NEUTRAL.
        # A model fine-tuned on SST-2 is typically very confident, so a slightly lower threshold
        # might better reflect genuine ambiguity vs. low confidence.
        neutral_threshold = 0.55 # Previously 0.6. Adjusted to make it less "neutral" by default.

        if score < neutral_threshold:
            return {'label': 'NEUTRAL', 'score': score}
        else:
            return {'label': label, 'score': score}

    except Exception as e:
        logging.error(f"Error during sentiment analysis for text: '{text[:50]}...': {e}")
        return {'label': 'NEUTRAL', 'score': 0.5} # Fallback for processing errors


def add_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sentiment label and score columns to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'review_text' column.

    Returns:
        pd.DataFrame: DataFrame with 'sentiment_label' and 'sentiment_score' columns.
    """
    if df.empty or 'review_text' not in df.columns:
        logging.warning("Input DataFrame for sentiment analysis is empty or missing 'review_text' column.")
        df['sentiment_label'] = 'NEUTRAL'
        df['sentiment_score'] = 0.5
        return df

    logging.info(f"Starting sentiment analysis for {len(df)} reviews...")
    
    # Filter out empty strings which can cause issues with some models
    df['review_text'] = df['review_text'].fillna('').astype(str).str.strip()
    
    # Apply sentiment analysis only to non-empty texts
    # Create temp columns to store results
    df['temp_sentiment_label'] = 'NEUTRAL'
    df['temp_sentiment_score'] = 0.5

    non_empty_texts_mask = df['review_text'].apply(lambda x: bool(x))
    
    if non_empty_texts_mask.any():
        # Apply sentiment analysis to non-empty texts using a list comprehension for efficiency
        # This approach is generally faster than .apply(lambda x: get_sentiment(x)) for large DFs
        texts_to_analyze = df.loc[non_empty_texts_mask, 'review_text'].tolist()
        
        # Chunking for very large datasets if needed, but pipeline handles batching
        sentiment_results = [get_sentiment(text) for text in texts_to_analyze]
        
        df.loc[non_empty_texts_mask, 'temp_sentiment_label'] = [res['label'] for res in sentiment_results]
        df.loc[non_empty_texts_mask, 'temp_sentiment_score'] = [res['score'] for res in sentiment_results]

    df['sentiment_label'] = df['temp_sentiment_label']
    df['sentiment_score'] = df['temp_sentiment_score']

    df.drop(columns=['temp_sentiment_label', 'temp_sentiment_score'], inplace=True)

    # KPI: Sentiment scores for 90%+ reviews
    # Count reviews that actually received a label other than the default neutral
    processed_reviews_count = df[df['sentiment_label'] != 'NEUTRAL']['review_text'].notna().sum()
    total_reviews = len(df)
    percentage_analyzed = (processed_reviews_count / total_reviews) * 100 if total_reviews > 0 else 0
    logging.info(f"Sentiment analysis complete. {percentage_analyzed:.2f}% of reviews processed (non-default sentiment).")
    if percentage_analyzed >= 90:
        logging.info("KPI met: Sentiment scores for 90%+ reviews (excluding default neutral).")
    else:
        logging.warning("KPI NOT met: Less than 90% of reviews processed for sentiment (excluding default neutral).")

    return df

if __name__ == "__main__":
    print("Running sentiment analysis module directly (for testing/debugging)...")
    dummy_data = {
        'review_text': ["This app is amazing! I love it.", "Absolutely terrible, crashes all the time.", "It's okay, nothing special.", "Login issue persists.", "", None, "Very good, highly recommended!"],
        'rating': [5, 1, 3, 2, 3, 4, 5],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07'],
        'bank': ['Bank A', 'Bank B', 'Bank A', 'Bank C', 'Bank A', 'Bank B', 'Bank A'],
        'reviewId': ['s1','s2','s3','s4','s5','s6','s7'],
        'userName': ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7'],
        'appVersion': ['1.0', '1.2', '1.0', '1.1', '1.0', '1.3', '1.0'],
        'source': ['Google Play Store'] * 7
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    sentiment_df = add_sentiment_scores(dummy_df)
    print("\nDataFrame with sentiment scores:")
    print(sentiment_df[['review_text', 'sentiment_label', 'sentiment_score']].to_string())