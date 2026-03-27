
![Tweeter](Images/Tweeter%20Image.png)

# Titan Electronic Sentiment Analysis

##  Project Overview
Titan Electronic Company operates in a competitive market selling high-end products such as iPhones, iPads, and other premium smartphones. Customer perception plays a critical role in influencing sales and brand loyalty.

This project leverages Natural Language Processing (NLP) to automatically classify customer sentiments from Twitter (X) data into **Positive**, **Negative**, or **Neutral** categories, enabling the company to gain real-time insights from large volumes of social media feedback.


##  Problem Statement
Titan Electronic receives thousands of tweets daily, making manual sentiment analysis inefficient and impractical. Additionally, tweets contain noisy text such as slang, abbreviations, and sarcasm, which complicates analysis.

The goal is to build a robust machine learning model capable of accurately classifying sentiment from this unstructured data.


##  Objectives
- Develop an automated sentiment classification model  
- Understand customer opinions and emotions  
- Monitor brand reputation in real time  
- Support data-driven marketing and product decisions  
- Handle large-scale tweet data efficiently  
 

## Approach
- Data collection from Twitter (X)  
- Data cleaning and preprocessing (handling noise, slang, etc.)  
- Feature extraction using NLP techniques  
- Model training and evaluation  
- Sentiment classification (Positive, Negative, Neutral)  


## Expected Impact
- Early detection of negative sentiment and potential PR issues  
- Identification of customer trends and preferences  
- Improved marketing strategies and decision-making  
- Enhanced customer satisfaction and brand positioning  


##  Tech Stack
- Python  
- NLP (NLTK, Scikit-learn, or Transformers)  
- Machine Learning Models (e.g., Logistic Regression, LSTM, BERT)  
- Data Visualization (Matplotlib, Seaborn)  


##  Conclusion
This project enables Titan Electronic to transform social media data into actionable insights, helping maintain a competitive edge while improving customer engagement and brand reputation.


# Customer Tweets Dataset on Tech Products

## Overview
This dataset contains tweets made by customers about **iPhone, Google, and Apple products**. Each row represents a single tweet and captures both the content and the sentiment directed toward the brand or product.

Understanding the dataset and its columns is crucial before any preprocessing or analysis.



## Dataset Columns

1. **`tweet text`**  
   - The actual text of the tweet.  
   - Contains customer opinions, comments, or reactions.  
   - May include emojis, hashtags, mentions, links, and informal language.  
   - Example:  
     
     "Loving the new iPhone camera! #AppleRocks "
     

2. **`emotion_in_tweet_is_directed_at`**  
   - The brand or product the tweet is about.  
   - Helps identify which product the sentiment is related to.  
   - Possible values: `iPhone`, `Google`, `Apple`, etc.  
   - Example:  
     
     "The Google Pixel battery life is amazing!" 
     

3. **`is_there_an_emotion_directed_at_a_brand_or_product`**  
   - The sentiment expressed in the tweet toward the brand/product.  
   - Values: `positive`, `negative`, `neutral`.  
   - Examples:  
     - Positive: "I love my new iPhone!" → `positive`  
     - Negative: "I hate how slow my iPhone is now!" → `negative`  
     - Neutral: "Apple released a new update for iOS." → `neutral`



 Notes for Preprocessing
- Tweets may contain informal text, typos, emojis, hashtags, mentions, or links.  
- Some tweets may mention multiple brands; check if the dataset handles this.  
- Neutral tweets can be tricky, especially if sarcasm is present.  
- Understanding the relationship between **tweet text**, **brand**, and **sentiment** is essential for any NLP task such as **sentiment analysis** or **brand-specific analysis**.


# Data Observations and Cleaning

We loaded the dataset `judge-1377884607_tweet_product_company.csv` and inspected its structure, shape, and data types.



## Shape and Data Types
- The dataset contains **9,093 rows** and **3 columns**:
  1. **`tweet text`** – contents of the tweet.
  2. **`emotion_in_tweet_is_directed_at`** – brand or product the tweet is about.
  3. **`is_there_an_emotion_directed_at_a_brand_or_product`** – sentiment of the tweet: `positive`, `negative`, or `neutral`.



## Missing Values
- **`tweet text`**: 1 missing value → will be **dropped**.  
- **`emotion_in_tweet_is_directed_at`**: 5,802 missing values (~64%) → will **not be dropped** because the brand/product information is optional and will not be used in modeling. Dropping over 60% would reduce dataset size and affect model training.  
- **`is_there_an_emotion_directed_at_a_brand_or_product`**: No missing values.



## Duplicates
- The dataset contains **27 duplicate rows** → will be **dropped** to prevent the model from learning repetitive data.



## Sentiment Labels
- The `is_there_an_emotion_directed_at_a_brand_or_product` column contains:  
  - `negative emotion`  
  - `positive emotion`  
  - `neutral emotion`  
- One ambiguous label, `"I can't tell"`, will be **dropped** because it does not provide a clear sentiment and could introduce noise that reduces model performance.



## Summary of Cleaning Steps
1. Drop **1 missing tweet text** row.  
2. Drop **27 duplicate rows**.  
3. Drop ambiguous sentiment label `"I can't tell"`.  
4. Keep `emotion_in_tweet_is_directed_at` column with missing values intact (optional metadata).  
5. Dataset is now ready for preprocessing and model training.


##  Data Preparation Steps

Prepare tweets for NLP modeling using these steps:

| Step | Description | Example |
|------|-------------|---------|
| ** Lemmatization** | Reduce words to their base form (lemma) | `studies` → `study` |
| ** Lowercasing** | Convert all text to lowercase to improve consistency and simplify processing | `Cats` → `cats` |
| ** Tokenization & Stopwords Removal** | Split sentences into words and remove common/irrelevant words | `"I love NLP"` → `["love", "NLP"]` |
| ** Label Encoding** | Convert categorical labels (text) into numeric form | `["positive","negative"]` → `[0,1]` |
| ** Stemming** | Remove suffixes to reduce words to their root | `cats` → `cat` |