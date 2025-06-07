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
    'DB_TYPE': 'postgresql', # Set to 'oracle' if you are using Oracle DB
    'DB_USER': 'your_db_user', # Replace with your database username
    'DB_PASSWORD': 'your_db_password', # Replace with your database password
    'DB_HOST': 'localhost',  # Replace with your database host (e.g., 'localhost', '192.168.1.100')
    'DB_PORT': '5432',       # Replace with your database port (e.g., '1521' for Oracle, '5432' for PostgreSQL)
    # For Oracle:
    'DB_SERVICE_NAME': 'XEPDB1', # Oracle Service Name (e.g., 'XEPDB1', 'XE', or SID).
                                 # This is usually required for Oracle connections.
    # For PostgreSQL: 'DB_NAME': 'bank_reviews_db' # Database name for PostgreSQL connection
}

# Other configurations (e.g., NLP model paths, thresholds) can be added here.