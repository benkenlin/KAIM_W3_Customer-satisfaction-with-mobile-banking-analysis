import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import re
# from collections import Counter # Not explicitly used in these functions, can be removed if not used elsewhere
import spacy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SpaCy model for tokenization and lemmatization (already in preprocessing, but for standalone run)
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("SpaCy model 'en_core_web_sm' loaded successfully for thematic analysis.")
except Exception as e:
    logging.error(f"Error loading SpaCy model in thematic analysis: {e}. Please ensure 'python -m spacy download en_core_web_sm' has been run.")


def extract_keywords_tfidf(df: pd.DataFrame, text_column: str = 'processed_text', top_n: int = 50) -> list:
    """
    Extracts top keywords using TF-IDF from the processed text column.

    Args:
        df (pd.DataFrame): DataFrame with a text column.
        text_column (str): Name of the column containing processed review text.
        top_n (int): Number of top keywords to extract.

    Returns:
        list: List of top keywords.
    """
    if df.empty or text_column not in df.columns or df[text_column].isnull().all():
        logging.warning(f"Input DataFrame for keyword extraction is empty or missing/empty '{text_column}' column.")
        return []

    logging.info("Extracting keywords using TF-IDF...")
    
    # Ensure text_column has no missing values and is string type
    corpus = df[text_column].fillna("").astype(str).tolist()
    
    # Filter out empty strings that might result from aggressive preprocessing
    corpus = [doc for doc in corpus if doc.strip()]

    if not corpus:
        logging.warning("Corpus is empty after filtering. No valid text data for TF-IDF keyword extraction.")
        return []

    try:
        # Removed stop_words='english' because preprocessing already handles it.
        # Adjusted min_df to 2: A word must appear in at least 2 documents to be considered.
        # This prevents very rare terms from skewing. You can lower it to 1 if needed.
        vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        # Get average TF-IDF score for each feature
        avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1 # .A1 converts to 1D numpy array
        sorted_indices = avg_tfidf_scores.argsort()[::-1] # Sort in descending order

        top_keywords = [feature_names[i] for i in sorted_indices[:top_n]]
        logging.info(f"Top {len(top_keywords)} keywords extracted.")
    except ValueError as e:
        # This specific ValueError "empty vocabulary" is caught here.
        logging.warning(f"Could not extract keywords (e.g., all docs too short, or no terms left after filtering): {e}")
        top_keywords = []
    except Exception as e:
        logging.error(f"An unexpected error occurred during TF-IDF keyword extraction: {e}")
        top_keywords = []
    return top_keywords

def assign_themes(df: pd.DataFrame, extracted_keywords: list, text_column: str = 'review_text') -> pd.DataFrame:
    """
    Assigns themes to reviews based on keyword presence. This is a rule-based
    approach using a predefined mapping of keywords to themes.

    Args:
        df (pd.DataFrame): DataFrame with 'review_text' column.
        extracted_keywords (list): List of keywords from TF-IDF or other methods.
                                   Used for informing theme definition, though not directly in matching.
        text_column (str): Name of the column containing the review text (original, not processed for matching).

    Returns:
        pd.DataFrame: DataFrame with 'identified_themes' column.
    """
    if df.empty or text_column not in df.columns:
        logging.warning("Input DataFrame for theme assignment is empty or missing text column.")
        df['identified_themes'] = 'Other' # Ensure column exists even if empty
        return df

    logging.info("Assigning themes to reviews...")

    # Define your themes and associated keywords/phrases.
    # THIS IS THE MOST IMPORTANT PART FOR THEMATIC ANALYSIS.
    # You MUST review the `extracted_keywords` and negative/positive reviews
    # to define these mappings accurately for your specific data.
    # This example is a starting point based on general banking app issues.
    theme_keywords = {
        'Account Access Issues': ['login', 'log in', 'access', 'password', 'face id', 'fingerprint', 'id', 'blocked', 'locked', 'cannot login'],
        'Transaction Performance': ['transfer', 'send money', 'receive money', 'transaction', 'payment', 'slow', 'delay', 'pending', 'speed', 'otp', 'fast', 'instant', 'withdrawal', 'deposit'],
        'User Interface & Experience': ['ui', 'interface', 'design', 'easy to use', 'user friendly', 'intuitive', 'clunky', 'confusing', 'experience', 'navigation', 'layout', 'lag'],
        'Customer Support': ['customer service', 'support', 'help', 'response', 'contact', 'call center', 'assist'],
        'Feature Requests': ['feature', 'add', 'option', 'new', 'dark mode', 'budgeting', 'bill payment', 'qr code'],
        'Bug/Crash/Performance': ['crash', 'bug', 'error', 'glitch', 'freezes', 'hangs', 'performance', 'issues', 'fix'],
        'Security Concerns': ['security', 'safe', 'secure', 'fraud', 'phishing', 'data', 'privacy', 'vulnerable']
    }

    # Convert keywords and review text to lowercase for case-insensitive matching
    lower_keywords = {theme: [kw.lower() for kw in kws] for theme, kws in theme_keywords.items()}
    
    # Initialize 'identified_themes' column
    df['identified_themes'] = ''

    # Process reviews and assign themes
    def map_themes(review_text):
        if not isinstance(review_text, str):
            return 'Other'
        
        text_lower = review_text.lower()
        found_themes = set()
        
        for theme, kws in lower_keywords.items():
            for kw in kws:
                # Use regex to match whole words or specific phrases more accurately
                # \b for word boundaries to avoid partial matches (e.g., 'bug' not 'debug')
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    found_themes.add(theme)
                    break # Move to next theme once a keyword for current theme is found
        
        if not found_themes:
            return 'Other'
        else:
            return ', '.join(sorted(list(found_themes)))

    df['identified_themes'] = df[text_column].apply(map_themes)
    logging.info("Themes assigned to reviews.")

    # Calculate and log theme distribution (optional for standalone run)
    # theme_counts = df['identified_themes'].str.split(', ').explode().value_counts().drop('').reset_index()
    # theme_counts.columns = ['Theme', 'Count']
    # logging.info(f"Overall Theme Distribution:\n{theme_counts.to_string()}")

    return df

if __name__ == "__main__":
    print("Running thematic analysis module directly (for testing/debugging)...")
    # Need to import tokenize_and_lemmatize if running directly
    from src.preprocessing import tokenize_and_lemmatize
    from datetime import date

    dummy_data = {
        'review_text': [
            "Login is super slow, crashes frequently. Fix your UI!",
            "Fast transfer but UI is confusing. Needs update.",
            "Great app, customer support is very responsive.",
            "Can't access my account, password reset not working.",
            "I wish they would add a budgeting feature. Good app though.",
            "This app is very secure and easy to use.",
            "My transactions are pending for days, horrible service.",
            "The dark mode feature would be awesome."
        ],
        'processed_text': [ # Preprocessed text for TF-IDF input
            "login super slow crash frequently fix ui",
            "fast transfer ui confuse need update",
            "great app customer support responsive",
            "access account password reset work",
            "wish add budgeting feature good app",
            "app secure easy use",
            "transaction pending day horrible service",
            "dark mode feature awesome"
        ],
        'bank': ['Bank A', 'Bank B', 'Bank A', 'Bank C', 'Bank A', 'Bank B', 'Bank C', 'Bank A']
    }
    dummy_df = pd.DataFrame(dummy_data)

    print("\nOriginal Dummy DataFrame:")
    print(dummy_df)

    # Extract keywords
    extracted_keywords = extract_keywords_tfidf(dummy_df, text_column='processed_text', top_n=20)
    print(f"\nExtracted Keywords: {extracted_keywords}")

    # Assign themes
    themed_df = assign_themes(dummy_df, extracted_keywords, text_column='review_text') # Pass original review_text for matching
    print("\nDummy DataFrame with Themes:")
    print(themed_df[['review_text', 'identified_themes']].to_string())

    # Check for empty 'identified_themes'
    empty_themes_count = themed_df[themed_df['identified_themes'].isnull() | (themed_df['identified_themes'] == '')].shape[0]
    print(f"\nReviews with empty/null themes: {empty_themes_count}")
    
    # Check if 'Other' is assigned correctly
    other_themes_count = themed_df[themed_df['identified_themes'] == 'Other'].shape[0]
    print(f"Reviews explicitly assigned 'Other': {other_themes_count}")