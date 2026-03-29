import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. LOAD YOUR NEW DATA
# Assuming you downloaded your Google Sheet as a CSV with 'Review' and 'Sentiment' columns
df = pd.read_csv("new_reviews.csv")

# Convert your text labels to numbers (Negative: 0, Positive: 1, Neutral: 2)
label_dict = {"Negative": 0, "Positive": 1, "Neutral": 2}
df['label'] = df['Sentiment'].map(label_dict)
df = df.dropna(subset=['label']) # Drop any blanks

# 2. LOAD YOUR EXISTING MODEL & TOKENIZER
model_name = "./titan_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 3. PREPARE THE DATASET
hf_dataset = Dataset.from_pandas(df[['Review', 'label']])

def tokenize_function(examples):
    return tokenizer(examples["Review"], padding="max_length", truncation=True)

tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

# 4. SET UP TRAINING RULES
training_args = TrainingArguments(
    output_dir="./titan_sentiment_model_v2",
    num_train_epochs=3,              # How many times to loop through the data
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# 5. TRAIN!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

print("Starting training...")
trainer.train()

# 6. SAVE THE UPGRADED MODEL
trainer.save_model("./titan_sentiment_model_v2")
tokenizer.save_pretrained("./titan_sentiment_model_v2")
print("New model saved to ./titan_sentiment_model_v2!")