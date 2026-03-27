
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

---

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