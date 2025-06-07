# KAIM_W3_Customer-satisfaction-with-mobile-banking-analysis
Analyzing customer satisfaction with mobile banking apps by collecting and processing user reviews from the Google Play Store for three Ethiopian banks: Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), Dashen Bank

# Mobile Banking App Review Analysis
## Project Overview
This project focuses on analyzing customer reviews for mobile banking applications available on the Google Play Store. Developed as part of a data analysis challenge for Omega Consultancy, the primary objective is to scrape, process, and analyze user feedback to identify satisfaction drivers and pain points, ultimately providing actionable recommendations to enhance mobile banking app features and user experience.

The analysis is structured to simulate real-world data engineering and analytics workflows, from data collection and robust preprocessing to advanced Natural Language Processing (NLP) for sentiment and thematic insights, secure data storage, and the generation of clear, data-driven recommendations.

## Business Objectives Addressed
(1) User Retention (Scenario 1): Identify and address issues causing user complaints, such as slow loading times during transfers, to improve user satisfaction and prevent churn.

(2) Feature Enhancement (Scenario 2): Extract desired features and functionalities from user feedback to inform app development and maintain competitive advantage.

(3) Complaint Management (Scenario 3): Cluster and track common complaints to guide the integration of AI chatbots and optimize customer support strategies for faster resolution.

## Key Features
=> Web Scraping: Automated collection of mobile banking app reviews, ratings, dates, and other metadata from the Google Play Store using google-play-scraper.

=> Data Preprocessing: Robust cleaning pipeline to handle missing values, remove duplicates, normalize dates, and clean raw text for NLP readiness.

=> Sentiment Analysis: Application of state-of-the-art NLP models (distilbert-base-uncased-finetuned-sst-2-english) to compute sentiment scores (positive, negative, neutral) for reviews.

=> Thematic Analysis: Identification of recurring themes and topics within reviews using TF-IDF and rule-based keyword matching to categorize user feedback into actionable areas (e.g., 'Account Access Issues', 'Transaction Performance', 'User Interface & Experience', 'Customer Support', 'Feature Requests', 'Security Concerns').

=> Database Storage: Design and implementation of a relational database schema (Oracle/PostgreSQL) to store cleaned and processed review data, simulating enterprise data engineering practices.

=> Insights Generation & Visualization: Derivation of actionable insights, comparison of performance across different banks, and creation of visualizations (sentiment distribution, sentiment by bank, themes by bank) to present findings.

=> Actionable Recommendations: Generation of specific, data-driven recommendations for app improvement based on identified pain points and satisfaction drivers.

=> Modular Design: Code structured into separate, reusable Python modules for clarity, maintainability, and scalability.

=> Version Control: Git integration with dedicated branches for each task, promoting frequent commits and pull request workflows.

## Project Structure
mobile_banking_reviews/
├── notebooks/
│   └── mobile_banking_analysis.ipynb       # Jupyter Notebook for interactive development and presentation
├── src/
│   ├── data_collection.py                  # Module for scraping reviews
│   ├── preprocessing.py                    # Module for data cleaning and normalization
│   ├── sentiment_analysis.py               # Module for sentiment scoring using DistilBERT
│   ├── thematic_analysis.py                # Module for keyword extraction and theme assignment
│   ├── database_manager.py                 # Module for database connection, schema, and CRUD operations
│   └── insights_generator.py               # Module for generating insights, visualizations, and recommendations
├── config/
│   ├── app_config.py                       # Stores application-specific configurations (App IDs, DB credentials)
│   └── db_schema.py                        # Defines SQL DDL for database tables (Banks, Reviews)
├── data/
│   ├── raw_reviews/                        # Directory to store initially scraped raw CSV data
│   └── processed_reviews/                  # Directory to store cleaned and processed CSV data
├── reports/
│   ├── sentiment_distribution.png          # Visualizations generated
│   ├── sentiment_by_bank.png
│   ├── themes_by_bank.png
│   └── recommendations.txt                 # Text file with actionable recommendations
├── .gitignore                              # Specifies intentionally untracked files to ignore
├── requirements.txt                        # Lists all Python dependencies
├── README.md                               # This file
└── pipeline.log                            # Log file for pipeline execution

## Setup Instructions
Follow these steps to set up and run the project:

### Clone the Repository (if applicable):

git clone <repository-url>
cd mobile_banking_reviews

### Create and Activate a Virtual Environment:
It's highly recommended to use a virtual environment to manage dependencies.

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

### Install Python Dependencies:
All required packages are listed in requirements.txt.

pip install -r requirements.txt

### Note for Database Drivers:

=> If DB_TYPE in config/app_config.py is set to 'oracle', ensure you have cx_Oracle installed. This often requires the Oracle Instant Client to be present on your system.

=> If DB_TYPE is set to 'postgresql', psycopg2-binary will be installed automatically from requirements.txt.

### Download SpaCy Language Model:
The en_core_web_sm model is used for text tokenization and lemmatization.

python -m spacy download en_core_web_sm

### Configure Application Settings (config/app_config.py):

=> BANK_APPS: Crucially, replace the placeholder App IDs (com.cbe.mobilebanking, etc.) with the actual Google Play Store App IDs for the mobile banking applications you intend to scrape. You can find these IDs in the URL of the app's page on Google Play (e.g., https://play.google.com/store/apps/details?id=YOUR_APP_ID).

=> DB_CONFIG: Update the database connection details (username, password, host, port, database name/service name) to match your Oracle or PostgreSQL setup. Remember to never hardcode sensitive credentials in production environments.

### Ensure Database is Running:
Make sure your Oracle or PostgreSQL database instance is running and accessible from your development environment.

## Usage
You can run the full data pipeline either as a standard Python script or interactively in a Jupyter Notebook.

### Option 1: Run as a Python Script (Full Pipeline)
This will execute all steps from data collection to insights generation in sequence.

python src/main.py

Output: Logs will be printed to the console and saved to pipeline.log. Processed data will be saved in data/processed_reviews/, and plots/recommendations in reports/.

### Option 2: Run Interactively in Jupyter Notebook
For step-by-step execution, exploration, and visualization:

Start Jupyter Notebook/Lab from the project's root directory:

jupyter notebook
# or jupyter lab

Navigate to the notebooks/ directory and open mobile_banking_analysis.ipynb.

Run the cells sequentially. The notebook is structured with Markdown headers for each task, allowing you to execute and review each stage of the pipeline interactively.

## Deliverables/Outputs
Upon successful execution, the following outputs will be generated:

data/raw_reviews/raw_bank_reviews.csv: The raw, scraped review data.

data/processed_reviews/processed_bank_reviews.csv: The cleaned and preprocessed review data.

Database Tables (banks, reviews): The cleaned data will be stored persistently in your configured Oracle/PostgreSQL database.

reports/sentiment_distribution.png: Bar chart showing the overall distribution of positive, neutral, and negative sentiments.

reports/sentiment_by_bank.png: Bar chart comparing the average sentiment score across different banks.

reports/themes_by_bank.png: Stacked bar chart illustrating the prevalence of identified themes for each bank.

reports/recommendations.txt: A text file containing actionable recommendations for each bank, based on sentiment and thematic analysis.

pipeline.log: A log file detailing the execution flow and any warnings/errors encountered during the pipeline run.

## Key Performance Indicators (KPIs)
The project aims to meet the following KPIs:

Data Collection: 1,200+ reviews collected with less than 5% missing data.

Data Quality: A clean CSV dataset (e.g., processed_bank_reviews.csv).

Sentiment Analysis: Sentiment scores for 90%+ of reviews.

Thematic Analysis: Identification of 3+ distinct themes per bank (excluding 'Other' category).

Code Quality: An organized Git repository with clear commit messages and a modular, well-documented codebase.

## Technologies Used
Python 3.x

google-play-scraper: For web scraping Google Play Store reviews.

pandas: For data manipulation and analysis.

scikit-learn: For TF-IDF vectorization.

transformers: For loading and using the DistilBERT sentiment analysis model.

torch (PyTorch): Backend for transformers.

spacy: For advanced text preprocessing (tokenization, lemmatization).

matplotlib, seaborn: For data visualization.

SQLAlchemy: Python SQL Toolkit and ORM for database interaction.

cx_Oracle (for Oracle) / psycopg2-binary (for PostgreSQL): Database drivers.

Git / GitHub: For version control.

Jupyter Notebook / JupyterLab: For interactive development and presentation.

## Author
Kenesa B. 

getkennyo@gmail.com
