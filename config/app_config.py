# app_config.py

# Web Scraping Configuration
BANK_APPS = {
    "CBE": "com.combanketh.mobilebanking",  # REPLACE WITH ACTUAL APP IDs
    "BOA": "com.boa.boaMobileBanking",
    "DashenBank": "com.dashen.dashensuperapp"
}
TARGET_REVIEWS_PER_BANK = 400

# Database Configuration (Oracle Example)
# IMPORTANT: Never hardcode sensitive credentials in production.
# Use environment variables or a secure configuration management system.
DB_CONFIG = {
    'DB_TYPE': 'oracle', # or 'postgresql'
    'DB_USER': 'system', # Your Oracle DB username (e.g., SYSTEM, HR, or custom user)
    'DB_PASSWORD': 'your_oracle_password', # Your Oracle DB password
    'DB_HOST': 'localhost', # Your Oracle DB host
    'DB_PORT': '1521',      # Your Oracle DB port (default for XE is 1521)
    'DB_SERVICE_NAME': 'XEPDB1', # For Oracle XE, often XEPDB1 or XE. Check your Listener.
    # For PostgreSQL: 'DB_NAME': 'bank_reviews_db' # Database name for PostgreSQL
}

# Add other configurations as needed (e.g., paths, model names)