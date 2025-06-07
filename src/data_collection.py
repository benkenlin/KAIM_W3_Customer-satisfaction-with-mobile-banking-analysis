import pandas as pd
from google_play_scraper import Sort, reviews_all
import logging
import time # Import time for adding delays
from config.app_config import BANK_APPS, TARGET_REVIEWS_PER_BANK

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_reviews(app_id: str, bank_name: str, max_count: int = TARGET_REVIEWS_PER_BANK) -> list:
    """
    Scrapes reviews for a given Google Play Store app ID.

    Args:
        app_id (str): The Google Play Store app ID.
        bank_name (str): The name of the bank associated with the app.
        max_count (int): The maximum number of reviews to scrape.

    Returns:
        list: A list of dictionaries, each representing a raw review.
    """
    logging.info(f"Attempting to scrape reviews for {bank_name} (App ID: {app_id})...")
    reviews = []
    try:
        # It's good practice to add a small delay to avoid being blocked,
        # especially if scraping many reviews.
        # sleep_milliseconds=0 is too aggressive for real-world scenarios.
        # Let's set it to 1000ms (1 second) as a default.
        scraped_reviews = reviews_all(
            app_id,
            sleep_milliseconds=1000,  # 1 second delay per request
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            filter_score_with=None,
            count=max_count # Use count parameter to limit directly if possible
        )

        # The `reviews_all` function returns a list, so we don't need to limit manually
        # if `count` parameter is used. If `count` is not used, manually slice.
        if len(scraped_reviews) > max_count:
             scraped_reviews = scraped_reviews[:max_count]

        for review in scraped_reviews:
            reviews.append({
                "content": review.get("content"),
                "score": review.get("score"),
                "at": review.get("at"),  # Datetime object
                "userName": review.get("userName"),
                "appVersion": review.get("appVersion", "N/A"),
                "reviewId": review.get("reviewId"),
                "bank": bank_name,
                "source": "Google Play Store"
            })
        logging.info(f"Successfully scraped {len(reviews)} reviews for {bank_name}.")
    except Exception as e:
        logging.error(f"Error scraping reviews for {bank_name}: {e}")
    return reviews

def collect_all_bank_reviews() -> pd.DataFrame:
    """
    Collects reviews for all configured banks and returns a DataFrame.
    """
    all_reviews = []
    for bank_name, app_id in BANK_APPS.items():
        bank_reviews = scrape_reviews(app_id, bank_name)
        all_reviews.extend(bank_reviews)
        # Add a delay between scraping different apps to be polite
        time.sleep(5) # 5 seconds delay between banks

    if not all_reviews:
        logging.warning("No reviews were collected. Check app IDs and network connection.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_reviews)
    logging.info(f"Total raw reviews collected across all banks: {len(df)}")
    return df

if __name__ == "__main__":
    import os
    print("Running data collection module directly (for testing/debugging)...")
    
    # Ensure data directory exists
    os.makedirs('data/raw_reviews', exist_ok=True)

    raw_df = collect_all_bank_reviews()
    if not raw_df.empty:
        raw_df.to_csv("data/raw_reviews/raw_bank_reviews.csv", index=False)
        print("Raw reviews saved to data/raw_reviews/raw_bank_reviews.csv")
    else:
        print("No raw data collected to save.")