import pandas as pd
import logging
import os
import sys

# Add the project root to the Python path to enable module imports
# Assuming main.py is in mobile_banking_reviews/src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import functions from other modules
from src.data_collection import collect_all_bank_reviews
from src.preprocessing import preprocess_reviews, tokenize_and_lemmatize
from src.sentiment_analysis import add_sentiment_scores
from src.thematic_analysis import extract_keywords_tfidf, assign_themes
from src.database_manager import DatabaseManager
from src.insights_generator import (
    generate_sentiment_summary,
    plot_sentiment_distribution,
    plot_sentiment_by_bank,
    plot_themes_by_bank,
    generate_recommendations
)
from config.app_config import DB_CONFIG

# Configure logging for the entire pipeline
log_file_path = 'pipeline.log'
# Remove previous log file for a fresh start if running directly
if os.path.exists(log_file_path):
    os.remove(log_file_path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])

def run_pipeline():
    """
    Orchestrates the entire mobile banking app review analysis pipeline.
    """
    logging.info("--- Starting Mobile Banking App Review Analysis Pipeline ---")

    # --- Step 1: Data Collection ---
    logging.info("\n[STEP 1/6] Starting Data Collection...")
    raw_df = collect_all_bank_reviews()
    if raw_df.empty:
        logging.error("Data collection failed or returned an empty DataFrame. Exiting pipeline.")
        return
    
    # Save raw data (optional, for debugging/record-keeping)
    os.makedirs('data/raw_reviews', exist_ok=True)
    raw_df.to_csv('data/raw_reviews/raw_bank_reviews.csv', index=False)
    logging.info(f"Raw data saved to data/raw_reviews/raw_bank_reviews.csv ({len(raw_df)} rows).")

    # --- Step 2: Preprocessing ---
    logging.info("\n[STEP 2/6] Starting Data Preprocessing...")
    processed_df = preprocess_reviews(raw_df.copy()) # Use a copy to avoid modifying original raw_df
    if processed_df.empty:
        logging.error("Preprocessing failed or returned an empty DataFrame. Exiting pipeline.")
        return
    
    # Save processed data
    os.makedirs('data/processed_reviews', exist_ok=True)
    processed_df.to_csv('data/processed_reviews/processed_bank_reviews.csv', index=False)
    logging.info(f"Processed data saved to data/processed_reviews/processed_bank_reviews.csv ({len(processed_df)} rows).")

    # --- Step 3: Sentiment Analysis ---
    logging.info("\n[STEP 3/6] Starting Sentiment Analysis...")
    sentiment_df = add_sentiment_scores(processed_df.copy())
    if sentiment_df.empty:
        logging.error("Sentiment analysis failed or returned an empty DataFrame. Exiting pipeline.")
        return
    
    # --- Step 4: Thematic Analysis ---
    logging.info("\n[STEP 4/6] Starting Thematic Analysis...")
    # The 'processed_text' column should be created in preprocessing.py for thematic analysis
    # If not, add a step here: sentiment_df['processed_text'] = sentiment_df['review_text'].apply(tokenize_and_lemmatize)
    if 'processed_text' not in sentiment_df.columns or sentiment_df['processed_text'].isnull().all():
        logging.warning("No 'processed_text' column or it's empty. Applying tokenization/lemmatization now for thematic analysis.")
        sentiment_df['processed_text'] = sentiment_df['review_text'].apply(tokenize_and_lemmatize)
    
    # Extract keywords based on processed text
    top_keywords = extract_keywords_tfidf(sentiment_df, text_column='processed_text', top_n=50)
    logging.info(f"Identified top keywords: {top_keywords[:10]}...")
    
    # Assign themes based on keywords (using original review_text for matching)
    final_analysis_df = assign_themes(sentiment_df.copy(), top_keywords, text_column='review_text')
    if final_analysis_df.empty:
        logging.error("Thematic analysis failed or returned an empty DataFrame. Exiting pipeline.")
        return

    # Check KPI: 3+ themes per bank
    # This check ensures that the thematic analysis is sufficiently granular.
    logging.info("Checking KPI: 3+ Themes per Bank for each bank with reviews.")
    for bank in final_analysis_df['bank'].unique():
        bank_themes = final_analysis_df[final_analysis_df['bank'] == bank]['identified_themes'].str.split(', ').explode().dropna().unique()
        num_themes = len([t for t in bank_themes if t != '' and t != 'Other']) # Exclude empty and 'Other'
        logging.info(f"Bank: {bank}, Themes identified (excluding 'Other'): {num_themes}")
        if num_themes >= 3:
            logging.info(f"  KPI met for {bank}: {num_themes} themes found.")
        else:
            logging.warning(f"  KPI NOT met for {bank}: Only {num_themes} themes found (less than 3). Review thematic analysis configuration.")


    # --- Step 5: Store Cleaned Data in Oracle (or PostgreSQL) ---
    logging.info("\n[STEP 5/6] Storing cleaned data in database...")
    db_manager = DatabaseManager(DB_CONFIG)
    data_for_insights = final_analysis_df # Default to current DataFrame if DB operation fails

    if db_manager.engine:
        db_manager.create_tables()
        db_manager.insert_banks_data(final_analysis_df) # Insert unique bank names first
        db_manager.insert_reviews_data(final_analysis_df) # Then insert reviews
        logging.info("Data storage attempted.") # Logging success/failure handled in db_manager
        
        # Always try to read data back from DB for insights to ensure consistency
        data_from_db = db_manager.read_reviews_from_db()
        if not data_from_db.empty:
            data_for_insights = data_from_db
            logging.info("Using data read from database for insights.")
        else:
            logging.warning("Failed to read data back from DB. Using the DataFrame before DB insertion for insights.")
    else:
        logging.error("Could not establish database connection. Skipping database operations and using DataFrame for insights.")

    # --- Step 6: Insights and Recommendations ---
    logging.info("\n[STEP 6/6] Generating Insights and Recommendations...")
    os.makedirs('reports', exist_ok=True) # Ensure reports directory exists

    if data_for_insights.empty:
        logging.error("No data available for generating insights and recommendations. Exiting pipeline.")
        return

    # Generate plots
    plot_sentiment_distribution(data_for_insights, save_path='reports/sentiment_distribution.png')
    plot_sentiment_by_bank(data_for_insights, save_path='reports/sentiment_by_bank.png')
    plot_themes_by_bank(data_for_insights, save_path='reports/themes_by_bank.png')
    logging.info("Plots generated and saved to 'reports/' directory.")

    # Generate and save text recommendations
    recommendations = generate_recommendations(data_for_insights)
    recommendations_file_path = 'reports/recommendations.txt'
    with open(recommendations_file_path, 'w', encoding='utf-8') as f:
        logging.info("\nActionable Recommendations:")
        for bank, rec_text in recommendations.items():
            logging.info(rec_text)
            f.write(rec_text + "\n\n")
    logging.info(f"Recommendations saved to {recommendations_file_path}")

    # Optional: Generate sentiment summary to console/log
    sentiment_summary_df = generate_sentiment_summary(data_for_insights)
    if not sentiment_summary_df.empty:
        logging.info("\nAggregated Sentiment Summary per Bank:\n" + sentiment_summary_df.to_string())

    logging.info("\n--- Pipeline Execution Complete! ---")

if __name__ == "__main__":
    run_pipeline()