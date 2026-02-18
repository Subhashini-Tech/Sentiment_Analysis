import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
#import sklearn
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk import word_tokenize
from spellchecker import SpellChecker
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.stem import PorterStemmer
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from collections import Counter
import base64
import pickle
import joblib
from saPreprocessing import preprocess_text
import warnings
warnings.filterwarnings("ignore")
#print(sklearn.__version__)    

# Upload the dataset
data = pd.read_csv("D:\Project5\chatgpt_style_reviews_dataset.csv")

# Required Columns - new data set
new_data = data[['review','rating']]

# Independent and Target
X = new_data['review']
y = new_data['rating']

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data set for EDA
c_data = data.drop(columns=['username', 'title'])
#c_data.head()
# Imputation - Date
c_data['date'] = c_data['date'].replace('########', np.nan)
c_data['date'] = c_data['date'].ffill().bfill()
# Preprocessing "Platform"
c_data['platform'] = c_data['platform'].replace(['Amazon', 'Flipkart', 'App Store', 'Google Play'], 'Mobile')

side = st.sidebar.radio(
    "Navigation",
    ["HOME", "SENTIMENT ANALYSIS", "INSIGHTS"])

# Back ground image
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# On choosing sidebar navigation
if side == "HOME":
    set_bg("D:\Project5\image4.jpg")
    st.markdown("""
    <h1 style="text-align: center; color: black">SENTIMENT ANALYSIS</h1>
    <h2 style="text-align: center; color: black;"> WELCOME!</h2>   
    <h3 style="text-align: center; color: black;">This application is designed to analyse the feedback given by customer.
                This helps to improve the functionality  of the application.</h3> 
    
    </style>               
    """,unsafe_allow_html=True)
if side == "SENTIMENT ANALYSIS":
    set_bg("D:\Project5\image4.jpg")
    st.markdown("""
    <h1 style="text-align: center; color: black">SENTIMENT ANALYSIS</h1>
        </style>               
    """,unsafe_allow_html=True)
    
    user_input = st.text_area("Enter your review")
    
    if st.button("Predict"):
        model = joblib.load(r"D:\\Project5\\nb_sentiment_model.pkl")
        processed = preprocess_text(user_input)
        prediction = model.predict([processed])[0]
        if prediction == 2 or prediction == "Positive":
            st.success("🟢 Positive")
        
        elif prediction == 0 or prediction == "Negative":
            st.error("🔴 Negative")
        
        elif prediction == 1 or prediction == "Neutral":
            st.warning("🟡 Neutral")
        proba = model.predict_proba([processed])[0]
        confidence = round(max(proba)*100, 2)
        st.info(f"Confidence: {confidence}%")

if side == "INSIGHTS":

    st.markdown("""
    <h1 style="text-align: center; color: black">INSIGHTS</h1>
    """, unsafe_allow_html=True)

    # Convert rating to sentiment
    sentiment_series = c_data['rating'].apply(
        lambda x: 'Negative' if x <= 2 else
                  'Neutral' if x == 3 else
                  'Positive'
    )

    # Count values in fixed order
    sentiment_counts = sentiment_series.value_counts().reindex(
        ['Negative', 'Neutral', 'Positive']
    )

    # Create figure properly
    fig1, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=['red', 'orange', 'green']  # Correct mapping
    )

    ax.set_title('Overall Sentiment Distribution')

    st.pyplot(fig1)

    st.markdown("""
    <h2 style="text-align:center; color:black;">
    Low Rating vs Sentiment Mismatch
    </h2>
    """, unsafe_allow_html=True)

    # Temporary sentiment mapping
    sentiment_series = c_data['rating'].apply(
    lambda x: 'Negative' if x <= 2 else
              'Neutral' if x == 3 else
              'Positive'
    )

    # Filter only 1-star and 2-star reviews
    low_rating_mask = c_data['rating'].isin([1, 2])
    low_rating_sentiments = sentiment_series[low_rating_mask]

    # Total count
    total_low_ratings = low_rating_sentiments.shape[0]

    # Mismatch count
    mismatch_count = low_rating_sentiments.isin(['Neutral', 'Positive']).sum()

    # Percentage
    mismatch_percentage = (
    (mismatch_count / total_low_ratings) * 100
    if total_low_ratings > 0 else 0
    )

    # Styled Output
    st.markdown(f"""
    <div style="background-color:#f0f2f6;
            padding:20px;
            border-radius:10px;
            text-align:center;">
            
    <h3>Total 1★ & 2★ Ratings: {total_low_ratings}</h3>
    <h3>Mismatch Count: {mismatch_count}</h3>
    <h3>Mismatch Percentage: {round(mismatch_percentage,2)}%</h3>

    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <h2 style="text-align:center; color:black;">
    WordCloud by Sentiment
    </h2>
    """, unsafe_allow_html=True)

    # Separate reviews by rating
    positive_reviews = c_data[c_data['rating'].isin([4, 5])]['review']
    negative_reviews = c_data[c_data['rating'].isin([1, 2])]['review']
    neutral_reviews  = c_data[c_data['rating'].isin([3])]['review']

    # Combine text
    positive_text = ' '.join(positive_reviews.dropna().astype(str))
    negative_text = ' '.join(negative_reviews.dropna().astype(str))
    neutral_text  = ' '.join(neutral_reviews.dropna().astype(str))

    # Generate word clouds
    pos_wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Greens'
    ).generate(positive_text)

    neg_wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Reds'
    ).generate(negative_text)

    neu_wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Blues'
    ).generate(neutral_text)

    # Create figure properly
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Positive
    axes[0].imshow(pos_wc, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Positive Reviews (4–5 Stars)', color='green')

    # Negative
    axes[1].imshow(neg_wc, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Negative Reviews (1–2 Stars)', color='red')

    # Neutral
    axes[2].imshow(neu_wc, interpolation='bilinear')
    axes[2].axis('off')
    axes[2].set_title('Neutral Reviews (3 Stars)', color='blue')

    plt.tight_layout()

    # Show in Streamlit
    st.pyplot(fig2)

    st.markdown("""
    <h2 style="text-align:center; color:black;">
    Monthly Sentiment Trend
    </h2>
    """, unsafe_allow_html=True)

    # Convert to datetime
    c_data['date'] = pd.to_datetime(c_data['date'])

    # Create time features
    c_data['year'] = c_data['date'].dt.year
    c_data['month'] = c_data['date'].dt.to_period('M')

    # Monthly counts
    monthly_counts = (
    c_data.groupby(['month', 'rating'])
          .size()
          .unstack(fill_value=0)
    )

    # Convert PeriodIndex to string (important for plotting)
    monthly_counts.index = monthly_counts.index.astype(str)

    # Create figure properly
    fig3, ax = plt.subplots(figsize=(12, 6))

    monthly_counts.plot(ax=ax, marker='o')

    ax.set_title("Monthly Sentiment Trend (Counts)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Reviews")
    ax.tick_params(axis='x', rotation=45)

    ax.legend(
    title="Rating",
    labels=[
        "1★ Very Negative",
        "2★ Negative",
        "3★ Neutral",
        "4★ Positive",
        "5★ Very Positive"
        ]
    )

    plt.tight_layout()

    # Show in Streamlit
    st.pyplot(fig3)


    st.markdown("""
    <h2 style="text-align:center; color:black;">
    High Ratings by Verified Purchase
    </h2>
    """, unsafe_allow_html=True)

    # Filter high ratings (4 & 5)
    high_ratings = c_data[c_data['rating'].isin([4, 5])]

    # Count verified purchases
    verified_counts = high_ratings['verified_purchase'].value_counts()

    # Create figure
    fig4, ax = plt.subplots(figsize=(6, 5))

    verified_counts.plot(
    kind='bar',
    ax=ax,
    color=['skyblue', 'orange']
    )

    # Add value labels on bars
    for i, count in enumerate(verified_counts.values):
        ax.text(i, count, str(count),
            ha='center',
            va='bottom',
            fontweight='bold')

    ax.set_xlabel('Verified Purchase')
    ax.set_ylabel('Count of 4 & 5 Ratings')
    ax.set_title('High Ratings by Verified Purchase')

    plt.tight_layout()

    # Show in Streamlit
    st.pyplot(fig4)

    st.markdown("""
    <h2 style="text-align:center; color:black;">
    Average Review Length by Sentiment
    </h2>
    """, unsafe_allow_html=True)

    # Calculate average review length per rating
    avg_length = (
        c_data.groupby('rating')['review_length']
          .mean()
          .reset_index()
        )

    # Create figure
    fig5, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
    data=avg_length,
    x='rating',
    y='review_length',
    palette='RdYlGn',
    ax=ax
    )

    ax.set_title("Average Review Length by Sentiment")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Average Review Length")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Show in Streamlit
    st.pyplot(fig5)


    st.markdown("""
    <h2 style="text-align:center; color:black;">
    Positive vs Negative Ratings by Location
    </h2>
    """, unsafe_allow_html=True)

    # Create flags
    c_data['positive'] = c_data['rating'].isin([4, 5])
    c_data['negative'] = c_data['rating'].isin([1, 2])

    # Aggregate by location
    location_counts = (
    c_data.groupby('location')
          .agg(
              positive_count=('positive', 'sum'),
              negative_count=('negative', 'sum')
          )
          .reset_index()
    )

    # Add total review count
    location_counts['total_reviews'] = (
    location_counts['positive_count'] + location_counts['negative_count']
    )

    # Filter locations with enough reviews
    location_counts = location_counts[location_counts['total_reviews'] > 20]

    # Prepare data for barplot
    plot_data = location_counts.melt(
    id_vars='location',
    value_vars=['positive_count', 'negative_count'],
    var_name='Sentiment_Type',
    value_name='Count'
    )

    # Sort by max sentiment
    plot_data = plot_data.sort_values(by='Count', ascending=False)

    # Create figure
    fig7, ax = plt.subplots(figsize=(12,6))

    sns.barplot(
    data=plot_data,
    x='location',
    y='Count',
    hue='Sentiment_Type',
    palette=['pink', 'blue'],
    ax=ax
    )

    ax.set_title("Positive (4&5) vs Negative (1&2) Ratings by Location")
    ax.set_xlabel("Location")
    ax.set_ylabel("Number of Reviews")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Sentiment", labels=['Positive (4&5)', 'Negative (1&2)'])

    plt.tight_layout()

    # Show in Streamlit
    st.pyplot(fig7)


    st.markdown("""
    <h2 style="text-align:center; color:black;">
    Number of Reviews by Platform
    </h2>
    """, unsafe_allow_html=True)

    # Count reviews per platform
    platform_counts = c_data.groupby('platform')['rating'].count()

    # Create figure
    fig8, ax = plt.subplots(figsize=(6, 5))

    platform_counts.plot(
    kind='bar',
    color=['skyblue', 'lightgreen'],
    ax=ax
    )

    # Labels and title
    ax.set_xlabel("Platform")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Number of Reviews by Platform")

    # Rotate x labels if needed
    ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()

    # Show in Streamlit
    st.pyplot(fig8)

    st.markdown("""
    <h2 style="text-align:center; color:black;">
    Highest vs Lowest Ratings by ChatGPT Version
    </h2>
    """, unsafe_allow_html=True)

    # Create flags
    c_data['high_rating'] = c_data['rating'].isin([4, 5])
    c_data['low_rating'] = c_data['rating'].isin([1, 2])

    # Aggregate by version
    version_counts = (
        c_data.groupby('version')
          .agg(
              high_count=('high_rating', 'sum'),
              low_count=('low_rating', 'sum'),
              total_reviews=('rating', 'count')
          )
          .reset_index()
        )       

    # Filter versions with enough reviews
    version_counts = version_counts[version_counts['total_reviews'] > 20]

    # Compute max impact for sorting
    version_counts['max_impact'] = version_counts[['high_count','low_count']].max(axis=1)
    version_counts = version_counts.sort_values(by='max_impact', ascending=False)

    # Melt for barplot
    plot_data = version_counts.melt(
    id_vars='version',
    value_vars=['high_count', 'low_count'],
    var_name='Rating_Type',
    value_name='Count'
    )

    # Create figure
    fig9, ax = plt.subplots(figsize=(10,6))

    sns.barplot(
    data=plot_data,
    x='version',
    y='Count',
    hue='Rating_Type',
    palette=['green', 'red'],
    ax=ax
    )

    # Labels and title
    ax.set_title("Highest (4&5) vs Lowest (1&2) Ratings by ChatGPT Version")
    ax.set_xlabel("ChatGPT Version")
    ax.set_ylabel("Number of Reviews")
    ax.legend(title="Rating", labels=['High (4&5)', 'Low (1&2)'])

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show in Streamlit
    st.pyplot(fig9)


    st.markdown("""
    <h2 style="text-align:center; color:black;">
    Most Common Negative Feedback Themes
    </h2>
    """, unsafe_allow_html=True)

    # --- Clean the review text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    c_data['clean_text'] = c_data['review'].apply(clean_text)
    negative_reviews = c_data[c_data['rating'].isin([1, 2])].copy()

    # --- Count vectorizer (optional, for reference)
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    vectorizer = CountVectorizer(stop_words='english', max_features=20)
    X = vectorizer.fit_transform(negative_reviews['clean_text'])

    word_counts = np.sum(X.toarray(), axis=0)
    words = vectorizer.get_feature_names_out()

    freq_df = pd.DataFrame({
    'word': words,
    'count': word_counts
    }).sort_values(by='count', ascending=False)

    # --- Define themes
    themes = {
    "Technical Errors": [
        "error", "errors", "bug", "bugs", "crash", "crashes",
        "issue", "issues", "problem", "problems", "not working",
        "failed", "failure"
    ],
    "Performance Issues": [
        "slow", "slower", "lag", "lagging", "delay",
        "response time", "takes long", "loading"
    ],
    "Accuracy Issues": [
        "wrong", "incorrect", "inaccurate", "bad answer",
        "nonsense", "irrelevant"
    ],
    "Login/Account Issues": [
        "login", "account", "sign in", "password", "access"
    ],
    "Subscription/Billing": [
        "refund", "charged", "charge", "payment",
        "subscription", "money"
    ],
    "Limited Features": [
        "limited", "limit", "restricted",
        "missing", "feature"
    ]
    }

    # --- Detect themes
    def detect_themes(text):
        matched = []
        for theme, keywords in themes.items():
            for word in keywords:
                if word in text:
                    matched.append(theme)
                    break
        return matched if matched else ["Other"]

    negative_reviews['detected_themes'] = negative_reviews['clean_text'].apply(detect_themes)

    # --- Count occurrences
    from collections import Counter

    all_themes = []
    for sublist in negative_reviews['detected_themes']:
        all_themes.extend(sublist)

    theme_counts = Counter(all_themes)

    # Ensure all themes appear
    for theme in themes.keys():
        if theme not in theme_counts:
            theme_counts[theme] = 0

    theme_df = pd.DataFrame(
    theme_counts.items(),
    columns=['Theme', 'Count']
    ).sort_values(by='Count', ascending=False)

    # Remove "Other"
    theme_df = theme_df[theme_df['Theme'] != "Other"]

    # --- Plot with Streamlit
    fig10, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
    data=theme_df,
    x='Count',
    y='Theme',
    palette='Reds_r',
    ax=ax
    )

    ax.set_title("Most Common Negative Feedback Themes")
    ax.set_xlabel("Number of Mentions")
    ax.set_ylabel("Theme")

    plt.tight_layout()

    st.pyplot(fig10)

