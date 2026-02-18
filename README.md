PROJECT: Sentiment Analysis
Description: This application analyzes user reviews of ChatGPT to determine overall sentiment and identify key factors influencing user satisfaction. The system evaluates textual feedback along with ratings. By detecting sentiment trends, version-based performance differences, and region-specific feedback patterns, the application helps uncover user experience insights, product improvement areas, and release impact analysis.

Technologies Used: Python, Streamlit, ML, EDA, NLP
Project Details:
Data Preparation & Pre-processing
• Collected and analyzed ChatGPT review dataset containing:
•	Review Text
•	Rating (1–5)
•	Version
•	Country
•	Verified User Status
•	Review Date
•	Review Length
•	Useful Review Count
• Cleaned dataset by:
•	Handling missing/null values
•	Removing duplicate reviews
•	Updated date column with backfill and forwardfill
•	Removing irrelevant fields
• Text Preprocessing steps applied:
•	Chat word expansion (e.g., LOL → Laughing Out Loud)
•	Lowercasing
•	Whitespace normalization
•	Tokenization
•	Spell correction
•	Stopword removal (while preserving sentiment words like not, very)
•	Punctuation removal
•	Lemmatization
• Sentiment Label Creation:
•	Ratings 1–2 → Negative
•	Rating 3 → Neutral
•	Ratings 4–5 → Positive
•	Applied Label Encoding for model training
• Feature Engineering:
•	Monthly and weekly sentiment aggregation
•	Version-based sentiment grouping
•	Country-level sentiment distribution
• Ensured consistent pre-processing pipeline for training and deployment.

EDA
Performed comprehensive EDA using Pandas and Seaborn to:
• Understand sentiment distribution across ratings
• Analyze review length vs sentiment relationship
• Identify peak dissatisfaction/satisfaction periods
• Detect country-based sentiment variations
• Examine version-based performance impact
• Identify most frequent negative feedback themes
Analyzed relationships between:
• Review length and sentiment score
• Version releases and user satisfaction
• Country and rating distribution
• Verified vs Non-verified user feedback patterns
• Useful review counts and sentiment polarity


Visualization
Built an interactive Streamlit dashboard to visualize:
• Overall sentiment distribution
• Sentiment trends over time (monthly/weekly)
• Version-wise sentiment comparison (clustered bar plots)
• Country-wise positive vs negative feedback
• Correlation heatmaps between numerical features
• Word clouds per sentiment class
• Negative feedback theme frequency charts
• Review length vs sentiment comparison
Machine Learning Models
Trained and evaluated multiple classification models:
•	Logistic Regression
•	Random Forest
•	Naive Bayes
•	LSTMs
•	BERT
•	Used TF-IDF vectorization for text feature extraction.
Model Evaluation Metrics:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
Model Selection & Deployment
• Selected best-performing model based on F1-score and AUC-ROC.
• Saved trained pipeline (TF-IDF + Naïve Bayes) using Pickle.
• Integrated the model into Streamlit application.

