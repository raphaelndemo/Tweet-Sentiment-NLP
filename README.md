
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
| Lemmatization | Reduce words to their base form (lemma) | `studies` → `study` |
|  Lowercasing | Convert all text to lowercase to improve consistency and simplify processing | `Cats` → `cats` |
|  Tokenization & Stopwords Removal | Split sentences into words and remove common/irrelevant words | `"I love NLP"` → `["love", "NLP"]` |
|  Label Encoding | Convert categorical labels (text) into numeric form | `["positive","negative"]` → `[0,1]` |
|  Stemming | Remove suffixes to reduce words to their root | `cats` → `cat` |


## Defining Feature and Target Variables

In this step, we specify the inputs and outputs for our NLP model:

 Feature (`X`): The input data used for prediction.  
  Typically, this is the preprocessed tweet text.

-Target (`y`): The output or label we want the model to predict.  
  For example, the sentiment class of each tweet (positive, negative, neutral).

  ## Class Distribution 

- Examine class distribution to identify imbalances before splitting the dataset.  
- Imbalanced classes can bias model evaluation if randomly split.  
- Apply **stratified sampling** during train/test split to preserve class proportions.  
- **80/20 train/test split** used.  
- **Fixed random state** ensures reproducibility.  

**Class distribution (%):**

| Sentiment Class                             | Percentage |
|---------------------------------------------|------------|
| No emotion toward brand or product          | 60.30%     |
| Positive emotion                             | 33.31%     |
| Negative emotion                             | 6.39%      |


## Modelling

The goal of this section is to build and compare classification models that predict sentiment from tweet text. **Pipelines** are used to ensure there is **no data leakage** during preprocessing and modeling.

We start with a **baseline model** and gradually introduce more complex models to improve performance.

### TF-IDF

- **TF (Term Frequency):** Counts how often each token appears in a tweet  
- **IDF (Inverse Document Frequency):** Measures the importance of each word relative to the entire corpus  
- TF-IDF converts raw text into **numerical feature vectors**, allowing models to understand which words are most relevant to the target.

### Baseline Model: Logistic Regression + TF-IDF

![Confusion Matrix Logistic Regression](Images/confusion%20matrix%20logistic%20regression.png)

- **Why Logistic Regression:** Simple, interpretable, and effective for text classification  
- **Role of TF-IDF:** Assigns weights to words, emphasizing important words and improving prediction accuracy  
- Models cannot understand raw text—they only work with numbers; TF-IDF ensures meaningful numerical representation  
- Without TF-IDF, all words are treated equally, which can reduce model performance


## Interpretation of Model Performance

### Negative Emotion
- **Precision:** 0.75  
- **Recall:** 0.05  
- **F1-score:** 0.62  

The model struggles with negative sentiment. While predictions labeled as negative are often correct (**high precision**), it rarely identifies negative cases (**very low recall**), causing most negative instances to be misclassified.



### Neutral Emotion
- **Precision:** 0.69  
- **Recall:** 0.87  
- **F1-score:** 0.77  

The model performs best on neutral sentiment. High recall indicates most neutral tweets are correctly identified, but lower precision shows frequent misclassification of positive and negative tweets as neutral, suggesting a bias toward this class.



### Positive Emotion
- **Precision:** 0.62  
- **Recall:** 0.44  
- **F1-score:** 0.51  

The model performs poorly on positive sentiment. Moderate precision with low recall indicates it fails to capture over half of actual positive tweets. Many positive instances are misclassified as neutral, highlighting a bias toward the neutral class and limiting detection of positive sentiment.

## Evaluation - Model Comparison

| Model | Precision | Recall | F1-score | Accuracy | Notes |
|-------|-----------|--------|----------|---------|-------|
| Logistic Regression + TF-IDF | 0.69 | 0.55 | 0.61 | 0.64 | Baseline model; moderate performance, biased toward neutral class |
| Naive Bayes + TF-IDF | 0.68 | 0.57 | 0.62 | 0.65 | Handles text well; slightly better recall than baseline |
| Random Forest | 0.71 | 0.60 | 0.65 | 0.67 | Higher precision; may overfit on small dataset |
| XGBoost | 0.73 | 0.62 | 0.67 | 0.69 | Best overall; balances precision and recall across classes |

**Key Points:**  
- Compare models using **Precision, Recall, F1-score, and Accuracy**  
- Examine performance across **positive, neutral, and negative** sentiments  
- Use results to select the **best-performing model** and guide optimization strategies


##  Tweaked Logistic Regression — Performance Analysis

###  Negative Emotion
- **Precision:** 0.33  
- **Recall:** 0.54  
- **F1-score:** 0.41  

The model demonstrates improved sensitivity to negative sentiment, achieving a recall of **0.54** with **62 out of 114** instances correctly identified. This indicates an enhanced ability to capture negative cases compared to the baseline.

However, the relatively low precision (**0.33**) reflects a high rate of false positives, where non-negative instances are incorrectly classified as negative. Analysis of the confusion matrix further reveals a tendency toward **over-prediction of the negative class**, with misclassifications distributed across both neutral and positive categories.



###  Neutral Emotion
- **Precision:** 0.78  
- **Recall:** 0.70  
- **F1-score:** 0.73  

The model achieves strong performance on the neutral class, maintaining a well-balanced trade-off between precision and recall. A total of **748 out of 1,074** neutral instances are correctly classified.

Nevertheless, some degree of class overlap persists, particularly with positive sentiment, indicating that the model encounters difficulty in distinguishing **subtle semantic differences** between neutral and adjacent classes.



###  Positive Emotion
- **Precision:** 0.58  
- **Recall:** 0.61  
- **F1-score:** 0.59  

Performance on the positive class is moderate, with **364 out of 594** instances correctly identified. While recall and precision are relatively balanced, the model exhibits notable misclassification into the neutral class.

This suggests a limitation in effectively separating **positive sentiment from neutral expressions**, likely due to overlapping linguistic features.



##  Overall Assessment

The tweaked Logistic Regression model exhibits a more **balanced sensitivity across sentiment classes**, particularly improving recall for negative sentiment. However, this improvement introduces trade-offs:

- Increased **false positive rate** for the negative class  
- Persistent **class overlap**, especially between neutral and positive sentiments  
- Reduced **precision in minority class predictions**  


## Model Interpretation: Naive Bayes with TF-IDF

### 1. Negative Emotion
- **Precision:** 0.83  
- **Recall:** 0.04  
- **F1-score:** 0.04  

The model achieves high precision but extremely low recall for negative sentiment. While predictions are accurate when made, it identifies only **5 out of 114** negative instances. This indicates the model is overly conservative and fails to capture most negative tweets.



### 2. Neutral Emotion
- **Precision:** 0.67  
- **Recall:** 0.89  
- **F1-score:** 0.77  

The model performs best on the neutral class, with a strong F1-score of 0.77. It correctly identifies **951 out of 1074** neutral instances. However, precision is moderate due to misclassification of non-neutral tweets—particularly **368 positive** and **90 negative** tweets incorrectly labeled as neutral.



### 3. Positive Emotion
- **Precision:** 0.62  
- **Recall:** 0.38  
- **F1-score:** 0.47  

The model shows moderate precision but low recall for positive sentiment, correctly identifying **226 out of 594** positive instances. A large number of positive tweets (**368**) are misclassified as neutral, suggesting difficulty distinguishing between positive and neutral sentiment.



### Summary
- The model is **biased toward predicting neutral sentiment**  
- It **misses most negative cases** despite high precision  
- It struggles to **differentiate positive from neutral**, leading to misclassification

## Trade-Off Analysis

- **Baseline Logistic Regression (TF-IDF):**  
  Performs poorly on negative sentiment with a recall of **0.05** and F1-score of **0.10**, indicating a strong bias toward the neutral class and minimal detection of negative tweets.

- **Baseline Naive Bayes (TF-IDF):**  
  Performs worse on negative sentiment, with a recall of **0.03** and F1-score of **0.05**. Despite high precision, it is largely ineffective at identifying negative feedback.

- **Tweaked Logistic Regression (TF-IDF):**  
  Shows the most improvement on negative sentiment, achieving a recall of **0.54** and F1-score of **0.41**. While it captures over half of negative cases, its lower precision (**0.33**) leads to more false positives.

- **Tweaked Naive Bayes (TF-IDF):**  
  Continues to underperform on negative sentiment with a recall of **0.04** and F1-score of **0.08**, remaining overly conservative and biased toward neutral predictions.

**Key Insight:**  
These results highlight a critical trade-off between **precision (operational efficiency)** and **recall (sensitivity to risk)**.


## Final Recommendation

### Tweaked Logistic Regression (TF-IDF)

This model is recommended for deployment because:

- It provides the best balance between **recall (0.54)** and **F1-score (0.41)** for negative sentiment detection.  
- It significantly improves the identification of negative feedback compared to other models.  
- It enables earlier detection of customer dissatisfaction, supporting proactive intervention.

**Business Impact:**  
Deploying this model will deliver the highest value to **Titan Electronic** by empowering teams across marketing, product, customer support, data analytics, public relations, sales, and management to address customer concerns before they escalate.

END.