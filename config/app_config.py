# config/app_config.py

# Web Scraping Configuration
# IMPORTANT: Replace these with actual Google Play Store App IDs for the banks you want to analyze.
# You can find the App ID in the Google Play Store URL (e.g., id=com.example.bankapp)
# Web Scraping Configuration
BANK_APPS = {
    "CBE": "com.combanketh.mobilebanking",  # REPLACE WITH ACTUAL APP IDs
    "BOA": "com.boa.boaMobileBanking",
    "DashenBank": "com.dashen.dashensuperapp"
}
TARGET_REVIEWS_PER_BANK = 400 # Minimum target reviews per bank

# Database Configuration (Choose 'oracle' or 'postgresql')
# IMPORTANT: NEVER hardcode sensitive credentials in production.
# Use environment variables (e.g., os.environ.get('DB_USER')) or a secure configuration management system.
DB_CONFIG = {
    'DB_TYPE': 'oracle',
    'DB_USER': 'BANK_USER',
    'DB_PASSWORD': '4567',
    'DB_HOST': 'localhost',
    'DB_PORT': '1521',
    'DB_SERVICE_NAME': 'xe', # Or 'XE' if using an older version/SID
}

# Other configurations (e.g., NLP model paths, thresholds) can be added here.