import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import Counter
import os # For ensuring directory exists

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set aesthetic style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100 # Adjust for higher resolution plots

def generate_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates average sentiment score and count per bank and sentiment label."""
    if df.empty:
        logging.warning("DataFrame is empty for sentiment summary.")
        return pd.DataFrame()
    
    # Ensure sentiment_label is a categorical type for consistent ordering
    sentiment_order = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    df['sentiment_label'] = pd.Categorical(df['sentiment_label'], categories=sentiment_order, ordered=True)

    sentiment_summary = df.groupby(['bank', 'sentiment_label'])['sentiment_score'].agg(['mean', 'count']).unstack(fill_value=0)
    
    # Flatten multi-level columns for cleaner output
    sentiment_summary.columns = [f"{col[0]}_{col[1]}" for col in sentiment_summary.columns]
    
    logging.info("Generated sentiment summary per bank.")
    return sentiment_summary

def plot_sentiment_distribution(df: pd.DataFrame, save_path: str = 'reports/sentiment_distribution.png'):
    """Plots the distribution of sentiment labels across all reviews."""
    if df.empty:
        logging.warning("DataFrame is empty for sentiment distribution plot.")
        return
    
    plt.figure(figsize=(8, 6))
    # Ensure categorical order for plotting
    sentiment_order = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    df['sentiment_label'] = pd.Categorical(df['sentiment_label'], categories=sentiment_order, ordered=True)

    sns.countplot(data=df, x='sentiment_label', palette='viridis', order=sentiment_order)
    plt.title('Overall Sentiment Distribution of Mobile Banking App Reviews', fontsize=14)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure directory exists
    plt.savefig(save_path)
    logging.info(f"Sentiment distribution plot saved to {save_path}")
    plt.close()

def plot_sentiment_by_bank(df: pd.DataFrame, save_path: str = 'reports/sentiment_by_bank.png'):
    """Plots the average sentiment score per bank."""
    if df.empty:
        logging.warning("DataFrame is empty for sentiment by bank plot.")
        return

    # Calculate mean sentiment score, excluding 'Neutral' for a clearer positive/negative comparison if desired
    # For now, let's include all to get a holistic average.
    avg_sentiment = df.groupby('bank')['sentiment_score'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_sentiment.index, y=avg_sentiment.values, palette='coolwarm')
    plt.title('Average Sentiment Score per Bank', fontsize=14)
    plt.xlabel('Bank', fontsize=12)
    plt.ylabel('Average Sentiment Score (0-1)', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logging.info(f"Sentiment by bank plot saved to {save_path}")
    plt.close()

def plot_themes_by_bank(df: pd.DataFrame, save_path: str = 'reports/themes_by_bank.png'):
    """Plots the prevalence of top themes per bank."""
    if df.empty or 'identified_themes' not in df.columns:
        logging.warning("DataFrame is empty or missing 'identified_themes' for themes by bank plot.")
        return
    
    # Explode themes to count them individually per review
    # Handle reviews with no identified themes (empty string or NaN)
    df_exploded = df.assign(theme=df['identified_themes'].str.split(', ')).explode('theme')
    df_exploded['theme'] = df_exploded['theme'].replace('', 'Other').fillna('Other')
    df_exploded = df_exploded[df_exploded['theme'] != ''] # Remove any empty strings from splitting

    if df_exploded.empty:
        logging.warning("No valid themes found for plotting themes by bank after exploding.")
        return

    # Count theme occurrences per bank
    theme_counts = df_exploded.groupby(['bank', 'theme']).size().unstack(fill_value=0)
    
    if theme_counts.empty:
        logging.warning("No themes found for plotting themes by bank.")
        return

    # Normalize counts to percentages within each bank for better comparison
    # Add a small epsilon to denominator to avoid division by zero if a bank has no reviews or themes
    theme_percentages = theme_counts.apply(lambda x: x / (x.sum() + 1e-9), axis=1)

    # Plot as stacked bar chart for better visualization of themes proportions
    fig, ax = plt.subplots(figsize=(12, 7))
    theme_percentages.plot(kind='bar', stacked=True, colormap='tab20', ax=ax)
    plt.title('Prevalence of Themes per Bank (Percentage of Reviews)', fontsize=14)
    plt.xlabel('Bank', fontsize=12)
    plt.ylabel('Percentage of Reviews', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Themes', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    logging.info(f"Themes by bank plot saved to {save_path}")
    plt.close()

def get_top_keywords_for_theme(df: pd.DataFrame, theme: str, text_column: str = 'processed_text', top_n: int = 5) -> list:
    """Extracts top keywords for a specific theme based on processed text."""
    # Filter reviews that contain the specified theme
    theme_reviews = df[df['identified_themes'].str.contains(theme, na=False)][text_column]
    
    if theme_reviews.empty:
        return []
    
    # Filter out empty strings from theme_reviews
    theme_reviews = theme_reviews[theme_reviews.astype(bool)]
    if theme_reviews.empty:
        return []

    # Use TF-IDF for keyword extraction
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1,2)) # No stop words as they are removed in preprocessing
    try:
        tfidf_matrix = vectorizer.fit_transform(theme_reviews.tolist())
        feature_names = vectorizer.get_feature_names_out()
        
        # Sum TF-IDF scores for each word across all reviews in the theme
        word_scores = tfidf_matrix.sum(axis=0).A1 # .A1 converts to 1D numpy array
        sorted_indices = word_scores.argsort()[::-1] # Sort in descending order
        
        top_keywords = [feature_names[i] for i in sorted_indices[:top_n]]
    except ValueError as e: # Catch cases where vectorizer might fail (e.g., all docs too short)
        logging.warning(f"Could not extract keywords for theme '{theme}': {e}")
        top_keywords = []
    
    return top_keywords

def generate_recommendations(df: pd.DataFrame) -> dict:
    """
    Generates actionable recommendations based on analysis.
    This function combines derived insights into textual recommendations.
    """
    if df.empty:
        logging.warning("DataFrame is empty for recommendation generation.")
        return {}

    recommendations = {}
    banks = df['bank'].unique()

    logging.info("Generating actionable recommendations...")

    for bank in banks:
        bank_df = df[df['bank'] == bank]
        total_reviews = len(bank_df)
        
        if total_reviews == 0:
            recommendations[bank] = f"For {bank}: No reviews available to generate recommendations."
            continue

        # Overall Sentiment
        pos_perc = (bank_df['sentiment_label'] == 'POSITIVE').sum() / total_reviews * 100
        neg_perc = (bank_df['sentiment_label'] == 'NEGATIVE').sum() / total_reviews * 100
        
        rec_text = f"--- Recommendations for {bank} ---\n"
        rec_text += f"Overall Sentiment: {pos_perc:.1f}% Positive, {neg_perc:.1f}% Negative Reviews.\n\n"

        # Pain Points (Negative Sentiment + Themes)
        neg_reviews_bank = bank_df[bank_df['sentiment_label'] == 'NEGATIVE']
        if not neg_reviews_bank.empty:
            themes_exploded_neg = neg_reviews_bank.assign(theme=neg_reviews_bank['identified_themes'].str.split(', ')).explode('theme')
            themes_exploded_neg['theme'] = themes_exploded_neg['theme'].replace('', 'Other').fillna('Other')
            themes_exploded_neg = themes_exploded_neg[themes_exploded_neg['theme'] != 'Other'] # Exclude 'Other'
            
            if not themes_exploded_neg.empty:
                pain_points_counts = themes_exploded_neg['theme'].value_counts()
                if not pain_points_counts.empty:
                    top_pain_point = pain_points_counts.index[0]
                    rec_text += f"Top Pain Point: '{top_pain_point}' ({pain_points_counts.iloc[0]} negative mentions).\n"
                    
                    keywords = get_top_keywords_for_theme(neg_reviews_bank, top_pain_point, 'processed_text', top_n=5)
                    if keywords:
                        rec_text += f"  Associated Keywords: {', '.join(keywords)}\n"
                    
                    if top_pain_point == 'Account Access Issues':
                        rec_text += "  Recommendation: Streamline login flows, improve multi-factor authentication (MFA) reliability, and enhance password recovery. Provide clear error messages.\n"
                    elif top_pain_point == 'Transaction Performance':
                        rec_text += "  Recommendation: Investigate and optimize backend processes for transfers and payments to reduce delays. Provide real-time transaction status updates.\n"
                    elif top_pain_point == 'User Interface & Experience':
                        rec_text += "  Recommendation: Prioritize fixing reported bugs and crashes. Conduct usability testing to improve navigation and overall app design.\n"
                    elif top_pain_point == 'Customer Support':
                        rec_text += "  Recommendation: Improve response times for customer inquiries. Explore AI-driven chatbots for instant answers to common questions.\n"
                    elif top_pain_point == 'Feature Requests':
                        rec_text += "  Recommendation: Evaluate highly requested features for future development. Conduct user surveys to validate demand and prioritize.\n"
                    elif top_pain_point == 'Security Concerns':
                        rec_text += "  Recommendation: Clearly communicate security features and protocols to users. Enhance fraud detection mechanisms and educate users on safe practices.\n"
                else:
                    rec_text += "No specific themed pain points identified from negative reviews.\n"
            else:
                rec_text += "No specific themed pain points identified from negative reviews.\n"
        else:
            rec_text += "Few to no negative reviews, focus on maintaining quality.\n"
        
        # Satisfaction Drivers (Positive Sentiment + Themes)
        pos_reviews_bank = bank_df[bank_df['sentiment_label'] == 'POSITIVE']
        if not pos_reviews_bank.empty:
            themes_exploded_pos = pos_reviews_bank.assign(theme=pos_reviews_bank['identified_themes'].str.split(', ')).explode('theme')
            themes_exploded_pos['theme'] = themes_exploded_pos['theme'].replace('', 'Other').fillna('Other')
            themes_exploded_pos = themes_exploded_pos[themes_exploded_pos['theme'] != 'Other'] # Exclude 'Other'

            if not themes_exploded_pos.empty:
                satisfaction_drivers = themes_exploded_pos['theme'].value_counts()
                if not satisfaction_drivers.empty:
                    top_driver = satisfaction_drivers.index[0]
                    rec_text += f"\nTop Satisfaction Driver: '{top_driver}' ({satisfaction_drivers.iloc[0]} positive mentions).\n"
                    keywords = get_top_keywords_for_theme(pos_reviews_bank, top_driver, 'processed_text', top_n=5)
                    if keywords:
                        rec_text += f"  Associated Keywords: {', '.join(keywords)}\n"
                    rec_text += f"  Recommendation: Highlight '{top_driver}' in marketing and continue to invest in improving these features.\n"
                else:
                    rec_text += "\nNo specific themed satisfaction drivers identified from positive reviews.\n"
            else:
                rec_text += "\nNo specific themed satisfaction drivers identified from positive reviews.\n"

        # General Recommendations
        rec_text += "\nGeneral Recommendation:\n"
        rec_text += "- Implement a continuous feedback loop: regularly scrape and analyze reviews to identify emerging trends.\n"
        rec_text += "- Prioritize development efforts based on the severity and frequency of reported pain points.\n"
        rec_text += "- Engage with users who leave critical reviews to understand their issues better and demonstrate responsiveness.\n"

        recommendations[bank] = rec_text.strip() + "\n" # Remove trailing newline

    logging.info("Recommendations generated.")
    return recommendations

if __name__ == "__main__":
    print("Running insights generator module directly (for testing/debugging)...")
    # Load dummy data or a sample from processed reviews
    from datetime import date
    dummy_data = {
        'review_text': [
            "Login is super slow, crashes frequently. Fix your UI!",
            "Fast transfer but UI is confusing. Needs update.",
            "Great app, customer support is very responsive.",
            "Can't access my account, password reset not working.",
            "I wish they would add a budgeting feature. Good app though.",
            "Smooth UI, easy to use, highly recommend.",
            "Transactions are fast, very reliable.",
            "Terrible support, never respond to emails. Very bad service.",
            "Fingerprint login is a pain, keeps failing. Security concern.",
            "Good design, but transfer limits are too low. Request new feature."
        ],
        'processed_text': [
            "login super slow crash frequently fix ui",
            "fast transfer ui confuse need update",
            "great app customer support responsive",
            "access account password reset work",
            "wish add budgeting feature good app",
            "smooth ui easy use highly recommend",
            "transaction fast reliable",
            "terrible support never respond email bad service",
            "fingerprint login pain keep fail security concern",
            "good design transfer limit low request new feature"
        ],
        'rating': [1, 3, 5, 1, 4, 5, 4, 2, 1, 3],
        'date': [date(2023,1,1), date(2023,1,2), date(2023,1,3), date(2023,1,4), date(2023,1,5),
                 date(2023,1,6), date(2023,1,7), date(2023,1,8), date(2023,1,9), date(2023,1,10)],
        'bank': ['Bank A', 'Bank B', 'Bank A', 'Bank C', 'Bank A', 'Bank A', 'Bank B', 'Bank C', 'Bank A', 'Bank B'],
        'sentiment_label': ['NEGATIVE', 'NEUTRAL', 'POSITIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE', 'NEGATIVE', 'NEGATIVE', 'NEUTRAL'],
        'sentiment_score': [0.95, 0.60, 0.98, 0.92, 0.75, 0.99, 0.88, 0.90, 0.96, 0.55],
        'identified_themes': [
            'Account Access Issues, User Interface & Experience', # Matches login, crashes, UI
            'Transaction Performance, User Interface & Experience', # Matches transfer, UI
            'Customer Support', # Matches customer support
            'Account Access Issues', # Matches access account, password reset
            'Feature Requests', # Matches budgeting feature
            'User Interface & Experience', # Matches UI
            'Transaction Performance', # Matches transactions fast
            'Customer Support', # Matches terrible support
            'Account Access Issues, Security Concerns', # Matches fingerprint login, security
            'Transaction Performance, Feature Requests' # Matches transfer, new feature
        ],
        'reviewId': [f'id{i}' for i in range(10)],
        'userName': [f'User{i}' for i in range(10)],
        'appVersion': ['1.0'] * 10,
        'source': ['Google Play Store'] * 10
    }
    dummy_df = pd.DataFrame(dummy_data)

    os.makedirs('reports', exist_ok=True) # Ensure reports directory exists

    sentiment_summary_df = generate_sentiment_summary(dummy_df)
    print("\nSentiment Summary:")
    print(sentiment_summary_df)

    plot_sentiment_distribution(dummy_df)
    plot_sentiment_by_bank(dummy_df)
    plot_themes_by_bank(dummy_df)

    recommendations_dict = generate_recommendations(dummy_df)
    print("\nRecommendations:")
    for bank, rec in recommendations_dict.items():
        print(rec)