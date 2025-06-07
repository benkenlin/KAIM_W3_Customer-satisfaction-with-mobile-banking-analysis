import sqlalchemy
import pandas as pd
import logging
from config.app_config import DB_CONFIG
from config.db_schema import BANKS_TABLE_SCHEMA, REVIEWS_TABLE_SCHEMA
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseManager:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.engine: Engine = None # Type hint for clarity
        self._connect()

    def _connect(self):
        """Establishes a connection to the database."""
        db_type = self.db_config.get('DB_TYPE', 'oracle').lower() # Normalize to lowercase
        conn_str = ""
        try:
            if db_type == 'oracle':
                # For Oracle, you need cx_Oracle.
                # Example connection string for Oracle XE:
                # "oracle+cx_oracle://system:password@localhost:1521/XEPDB1"
                # Ensure username, password, host, port, and service_name (or SID) are correct.
                # If using TNS names: 'oracle+cx_oracle://user:pass@tns_name'
                conn_str = (
                    f"oracle+cx_oracle://{self.db_config['DB_USER']}:{self.db_config['DB_PASSWORD']}@"
                    f"{self.db_config['DB_HOST']}:{self.db_config['DB_PORT']}/{self.db_config['DB_SERVICE_NAME']}"
                )
                logging.info(f"Attempting to connect to Oracle database at {self.db_config['DB_HOST']}:{self.db_config['DB_PORT']}...")
            elif db_type == 'postgresql':
                # For PostgreSQL, you need psycopg2-binary.
                conn_str = (
                    f"postgresql+psycopg2://{self.db_config['DB_USER']}:{self.db_config['DB_PASSWORD']}@"
                    f"{self.db_config['DB_HOST']}:{self.db_config['DB_PORT']}/{self.db_config['DB_NAME']}"
                )
                logging.info(f"Attempting to connect to PostgreSQL database at {self.db_config['DB_HOST']}:{self.db_config['DB_PORT']}...")
            else:
                logging.error(f"Unsupported database type: {db_type}. Please configure 'DB_TYPE' in app_config.py to 'oracle' or 'postgresql'.")
                return

            self.engine = create_engine(conn_str)
            # Test connection
            with self.engine.connect() as connection:
                if db_type == 'oracle':
                    connection.execute(text("SELECT 1 FROM DUAL"))
                elif db_type == 'postgresql':
                    connection.execute(text("SELECT 1"))
            logging.info(f"Successfully connected to {db_type} database.")
        except ImportError as ie:
            logging.error(f"Required DB driver not installed for {db_type}: {ie}. Please install it (e.g., 'pip install cx_Oracle' or 'pip install psycopg2-binary').")
            self.engine = None
        except SQLAlchemyError as se:
            logging.error(f"SQLAlchemy error connecting to {db_type} database: {se}")
            self.engine = None
        except Exception as e:
            logging.error(f"An unexpected error occurred during {db_type} connection: {e}")
            self.engine = None

    def create_tables(self):
        """Creates the Banks and Reviews tables if they don't exist."""
        if self.engine is None:
            logging.error("Database engine not initialized. Cannot create tables.")
            return

        inspector = inspect(self.engine)
        
        with self.engine.connect() as connection:
            try:
                # Get the appropriate schema for the DB type
                db_type = self.db_config.get('DB_TYPE', 'oracle').lower()
                banks_schema_ddl = BANKS_TABLE_SCHEMA.get(db_type)
                reviews_schema_ddl = REVIEWS_TABLE_SCHEMA.get(db_type)

                if not banks_schema_ddl or not reviews_schema_ddl:
                    logging.error(f"Database schema DDL not defined for type: {db_type}")
                    return

                # Create Banks table
                if not inspector.has_table('banks'):
                    logging.info("Creating 'banks' table...")
                    connection.execute(text(banks_schema_ddl))
                    logging.info("'banks' table created successfully.")
                else:
                    logging.info("'banks' table already exists.")

                # Create Reviews table
                if not inspector.has_table('reviews'):
                    logging.info("Creating 'reviews' table...")
                    connection.execute(text(reviews_schema_ddl))
                    logging.info("'reviews' table created successfully.")
                else:
                    logging.info("'reviews' table already exists.")
                connection.commit() # Commit table creation DDL
            except SQLAlchemyError as e:
                connection.rollback()
                logging.error(f"Error creating tables: {e}")
            except Exception as e:
                connection.rollback()
                logging.error(f"An unexpected error occurred during table creation: {e}")


    def insert_banks_data(self, banks_df: pd.DataFrame):
        """Inserts unique bank names into the Banks table."""
        if self.engine is None or banks_df.empty:
            logging.error("Database engine not initialized or banks DataFrame is empty. Cannot insert bank data.")
            return

        unique_banks = banks_df['bank'].unique()
        banks_to_insert = pd.DataFrame({'bank_name': unique_banks})

        try:
            logging.info(f"Inserting {len(banks_to_insert)} unique banks into 'banks' table.")
            # Use 'append' mode and handle duplicates.
            # On conflict, 'DO NOTHING' is a PostgreSQL feature. For Oracle, you'd use MERGE or check existence.
            # A simpler way for general purpose is to load existing banks and filter.
            with self.engine.connect() as connection:
                existing_banks = pd.read_sql_table('banks', connection).bank_name.tolist()
                new_banks_df = banks_to_insert[~banks_to_insert['bank_name'].isin(existing_banks)]
                if not new_banks_df.empty:
                    new_banks_df.to_sql('banks', self.engine, if_exists='append', index=False)
                    logging.info(f"Inserted {len(new_banks_df)} new banks into 'banks' table.")
                else:
                    logging.info("No new banks to insert.")
        except SQLAlchemyError as e:
            logging.error(f"Error inserting banks data: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during bank data insertion: {e}")

    def insert_reviews_data(self, reviews_df: pd.DataFrame):
        """
        Inserts processed review data into the Reviews table.
        Assumes 'review_id', 'review_text', 'rating', 'date', 'bank', 'sentiment_label',
        'sentiment_score', 'identified_themes', 'user_name', 'app_version', 'source' columns exist.
        """
        if self.engine is None or reviews_df.empty:
            logging.error("Database engine not initialized or reviews DataFrame is empty. Cannot insert review data.")
            return
        
        # Ensure 'date' is converted to appropriate format for DB
        # Convert to datetime, then to date if needed for SQL DATE type
        reviews_df['date'] = pd.to_datetime(reviews_df['date']).dt.date

        # To link reviews to banks, we need bank_id.
        # This requires fetching bank_ids from the 'banks' table.
        try:
            with self.engine.connect() as connection:
                existing_banks_df = pd.read_sql_table('banks', connection)
                
                # Merge reviews_df with existing_banks_df to get bank_id
                # Make sure 'bank' column in reviews_df matches 'bank_name' in banks table
                reviews_df_with_bank_id = pd.merge(
                    reviews_df,
                    existing_banks_df,
                    left_on='bank',
                    right_on='bank_name',
                    how='left'
                )
                
                # Drop rows where bank_id couldn't be found (shouldn't happen if banks are inserted first)
                reviews_df_with_bank_id.dropna(subset=['bank_id'], inplace=True)
                reviews_df_with_bank_id['bank_id'] = reviews_df_with_bank_id['bank_id'].astype(int)

                # Select and rename columns that match the 'reviews' table schema
                final_df_for_db = reviews_df_with_bank_id[[
                    'reviewId', 'review_text', 'rating', 'date', 'bank_id',
                    'sentiment_label', 'sentiment_score', 'identified_themes',
                    'userName', 'appVersion', 'source'
                ]].copy()
                
                # Rename columns to match DB schema exactly if needed
                final_df_for_db.rename(columns={
                    'reviewId': 'review_id',
                    'date': 'review_date',
                    'userName': 'user_name',
                    'appVersion': 'app_version'
                }, inplace=True)

                logging.info(f"Inserting {len(final_df_for_db)} reviews into 'reviews' table.")
                # Use 'append' mode and handle duplicates by relying on the 'review_id' UNIQUE constraint in the DB.
                # Pandas' to_sql doesn't have native ON CONFLICT for all DBs.
                # For Oracle, you'd typically need a MERGE statement or pre-check existence.
                # For simplicity with to_sql, we'll try insert and catch unique constraint errors if any.
                try:
                    final_df_for_db.to_sql('reviews', self.engine, if_exists='append', index=False)
                    logging.info("Review data inserted successfully.")
                except SQLAlchemyError as se:
                    if "unique constraint" in str(se).lower() or "duplicate key" in str(se).lower():
                        logging.warning(f"Some reviews already exist (duplicate review_id). Skipping existing rows. Error: {se}")
                        # You might implement more robust UPSERT logic here for production
                    else:
                        raise # Re-raise other SQLAlchemy errors
        except SQLAlchemyError as e:
            logging.error(f"Error during reviews data insertion: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during reviews data insertion: {e}")

    def read_reviews_from_db(self) -> pd.DataFrame:
        """Reads all reviews from the database."""
        if self.engine is None:
            logging.error("Database engine not initialized. Cannot read from DB.")
            return pd.DataFrame()
        try:
            logging.info("Reading reviews from database...")
            # Using raw SQL to join tables and get bank_name directly
            query = """
            SELECT
                r.review_id,
                r.review_text,
                r.rating,
                r.review_date AS date, -- Alias to 'date' for consistency with DataFrames
                b.bank_name AS bank,   -- Alias to 'bank'
                r.sentiment_label,
                r.sentiment_score,
                r.identified_themes,
                r.user_name,
                r.app_version,
                r.source
            FROM reviews r
            JOIN banks b ON r.bank_id = b.bank_id
            """
            df = pd.read_sql(text(query), self.engine)
            
            # Convert 'date' column back to datetime object (if needed for analysis later)
            df['date'] = pd.to_datetime(df['date'])

            logging.info(f"Successfully read {len(df)} reviews from the database.")
            return df
        except SQLAlchemyError as e:
            logging.error(f"Error reading reviews from database: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"An unexpected error occurred during database read: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    print("Running database manager module directly (for testing/debugging)...")
    # Make sure your config/app_config.py and config/db_schema.py are set up
    # before running this directly.
    
    from datetime import date
    dummy_processed_data = {
        'reviewId': ['r1','r2','r3','r4','r5', 'r6'],
        'review_text': ["App is great, very smooth.", "Crashes a lot, bad UI.", "Login issue fixed, finally!", "Good support.", "Slow transfers.", "Another good app from Bank A"],
        'rating': [5, 1, 4, 5, 2, 4],
        'date': [date(2024,1,1), date(2024,1,2), date(2024,1,3), date(2024,1,4), date(2024,1,5), date(2024,1,6)],
        'bank': ['Bank A', 'Bank B', 'Bank A', 'Bank C', 'Bank B', 'Bank A'],
        'sentiment_label': ['POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'POSITIVE'],
        'sentiment_score': [0.95, 0.98, 0.85, 0.92, 0.91, 0.93],
        'identified_themes': ['User Interface & Experience', 'User Interface & Experience, Transaction Performance', 'Account Access Issues', 'Customer Support', 'Transaction Performance', 'User Interface & Experience'],
        'userName': ['U1','U2','U3','U4','U5','U6'],
        'appVersion': ['1.0','1.2','1.0','1.1','1.2','1.0'],
        'source': ['Google Play Store'] * 6
    }
    dummy_df = pd.DataFrame(dummy_processed_data)

    db_manager = DatabaseManager(DB_CONFIG)
    if db_manager.engine:
        db_manager.create_tables()
        db_manager.insert_banks_data(dummy_df) # Insert banks first
        db_manager.insert_reviews_data(dummy_df) # Then insert reviews
        
        print("\nReading data from DB...")
        df_from_db = db_manager.read_reviews_from_db()
        print(df_from_db.head())
        print(f"Total rows read from DB: {len(df_from_db)}")
    else:
        print("Could not connect to database. Check configurations and ensure Oracle/PostgreSQL is running.")