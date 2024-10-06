
# BERT Mediated Sentiment Analysis on IMDb Reviews: Rule-Based Detection of Mixed Reviews Using Sentence Splitting 🤖

> A NLP tutorial on fine-tuning the pre-trained **BERT** model on the IMDb review dataset includes positive and negative labels.  
> After the model achieved great performance on test data, I implemented function leveraging the output logits from BERT and applying threshold-based logic for classification into 3 classes: positive, negative, and mixed.   
> Ultimately, this work proves that fine-tuning transformer-based models is essential for sentiment analysis in today's data-driven world.

**Project exemplifies codes on how to perform the following tasks:**  
1. Coding environment preparation (e.g., setting global seeds for reproducibility)
2. Downloading the IMDb review dataset using the `Hugging Face` library
3. Loading and working with BERT Tokenizer (via `Hugging Face`)
4. Loading pre-trained BERT model and setting device for model (via `Hugging Face`)
5. Dataset processing: splitting train/test, tokenizing, converting to torch tensors
6. Defining optimizer and learning rate schedular
7. Running training loop for BERT on training dset
8. Saving & Loading fine-tuned BERT model 
9. Defining the evaluation loop to get model predictions
10. Assessing model performance by classification metrics and confusion matrix
11. Implementing custom function for rule-based detection approach (see *`Model Performance`*)
11.1. Sentence the review: splitting the review into individual sentences  
11.2. Sentiment analysis of each sentence  
11.3. Threshold logic to determine mixed reviews

🎯 **Prediction Output of Rule-Based Detection Approach** 🎯
[<img src="https://github.com/SevilayMuni/BERT-SentimentAnalysis-IMDbReviews/blob/main/images/example-reviews-mixed-label.png" width="800"/>](https://github.com/SevilayMuni/BERT-SentimentAnalysis-IMDbReviews/blob/main/images/example-reviews-mixed-label.png)

## Model Introduction 📚

I utilized the BERT (Bidirectional Encoder Representations from Transformers) model, specifically the bert-base-uncased version from Hugging Face's library.    
BERT leverages transformers, which allow the model to capture the context of a word in a sentence by looking at both the preceding and following words (bidirectional training). This is crucial for understanding the nuances in natural language, especially in tasks like sentiment analysis where word order and context significantly impact meaning.

> Compared to traditional machine learning models, BERT offers several advantages:  
> 
>> ✅ Contextual embeddings: BERT generates different embeddings based on the word's context within a sentence.  
>> ✅ Pre-trained on a large corpus: BERT is trained on a vast amount of data (e.g., Wikipedia, BooksCorpus), allowing it to develop a deep understanding of language.   
>> ✅ Fine-tuning: BERT requires minimal task-specific efforts for tasks like classification, making it adaptable and efficient for a wide range of NLP problems.
## Data and Processing 👩🏼‍💻

IMDb Movie Reviews Dataset contains 100,000 movie reviews. The dataset is balanced with positive and negative reviews, and duplicates were removed to ensure data quality. 

**Data Processing Steps:**  
    *1. Data Preprocessing*
- Dataset Splitting: The dataset was split into training/testing sets to evaluate the model's performance on unseen data during training.
- Tokenization: Using the BertTokenizer from Hugging Face's transformers library, the text reviews were tokenized into subword tokens that BERT could process. The tokenizer splits the input text into tokens of a fixed maximum sequence length (512) and pads or truncates them as needed. Each review was converted into:  
    - Input IDs: Tokenized words represented as numerical IDs.
    - Attention Masks: A binary mask that differentiates between real tokens (1) and padding tokens (0).

```
# Function to preprocess the text into tokens
def preprocess_function(examples):
    return tokenizer(examples['text'],
                     padding = True,
                     truncation = True,
                     return_tensors = 'pt')

```

*2. Converting Data to PyTorch Tensors*
- Tensors are compatible with PyTorch Models
- PyTorch tensors enables batch processing with DataLoader.
- Tensors are the primary data structure that can be transferred to a GPU (cuda()) for accelerated training.

```
def format_dataset(batch):
    # Convert the labels and the inputs to PyTorch tensors
    batch['label'] = torch.tensor(batch['label'])
    batch['input_ids'] = torch.tensor(batch['input_ids'])
    batch['attention_mask'] = torch.tensor(batch['attention_mask'])
    return batch
```
*3. Creating Data Loaders:*
- DataLoader objects handle batching and shuffling of the data during training. 
- It allows for efficient feeding of data to the model in batches during the training loop.
## Model Hyperparameters 📍
> Optimizer: AdamW  
> Learning Rate: 5e-5 with linear decay   
> Batch Size: 8  
> Epochs: 3  

```
# Define the optimizer
optimizer = AdamW(model.parameters(), lr = 5e-5)

# Define the learning rate scheduler
num_training_steps = len(train_dataloader) * 3 # 3 epochs
lr_scheduler = get_scheduler('linear',
                             optimizer = optimizer,
                             num_warmup_steps = 0,
                             num_training_steps = num_training_steps)
```
## Model Performance 🎯

The model was trained on a subset of the IMDb data. The model performance was examined by confusion matrix and evaluation metrics; precision, recall, and F1-score.

###  Classification Metrics Table

| Class | Precision | Recall | F1-Score | 
| --- | --- | --- | --- |
| `Negative` | 0.93 | 0.91 | 0.92 |
| `Positive` | 0.91 | 0.92 | 0.91 |
| `Accuracy` | |  | 0.92 |
| `Macro Avg.` | 0.92| 0.92| 0.92 |
| `Micro Avg.` | 0.92| 0.92 | 0.92|

###  Confusion Matrix
[<img src="https://github.com/SevilayMuni/BERT-SentimentAnalysis-IMDbReviews/blob/main/images/confusion-matrix.png" width="500"/>]([https://github.com/SevilayMuni/BERT-SentimentAnalysis-IMDbReviews/blob/main/images/example-reviews-mixed-label.png](https://github.com/SevilayMuni/BERT-SentimentAnalysis-IMDbReviews/blob/main/images/confusion-matrix.png))

###  Rule-Based Detection Using Sentence Splitting
Customized logit thresholds were used for the multi-class sentiment (positive, negative, mixed).

```
# Define function to classify reviews into 3 classes based on logits
def classify_review_mixed(review):
    # Tokenize the review into sentences
    sentences = nltk.sent_tokenize(review)
    pos_count = 0
    neg_count = 0

    # Classify each sentence
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors = 'pt', 
                           padding = True, truncation = True, 
                           max_length = 512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get the prediction (0: Negative, 1: Positive)
            # Convert tensor to scalar
            prediction = torch.argmax(logits, dim = -1).item()

        # Increment counts based on prediction
        if prediction == 1:
            pos_count += 1

        else:
            neg_count += 1

    # Threshold logic to determine 'Mixed'
    if pos_count > 0 and neg_count > 0:
        return 'Mixed'
    elif pos_count > 0:
        return 'Positive'
    else:
        return 'Negative'
```

## Conclusion 📝

📌 This project successfully demonstrates the power and flexibility of BERT for sentiment analysis on IMDb movie reviews.  
📌 By fine-tuning a pre-trained BERT model, the model achieved high accuracy in classifying reviews into positive and negative categories.  
📌 To extend this further, I implemented a custom approach to classify reviews into three categories—positive, negative, and mixed—by leveraging the output logits from BERT and applying threshold-based logic. 

🔹 With BERT's ability to capture context from both directions in a sentence, it stands out as a valuable tool for any sentiment analysis task.  
🔹 This project highlights BERT's effectiveness in handling text classification tasks. It sets the foundation for more complex NLP tasks such as **opinion mining**, **review summarization**, or other business intelligence and customer feedback analysis applications.

## Future Insights 

The project can be further extended by experimenting with *larger datasets*, *exploring different pre-trained models*, or even *integrating more fine-grained sentiment analysis* for better-nuanced understanding. 

## Modules & Libraries 🗂️
matplotlib==3.7.2  
nltk==3.8.1  
numpy==1.23.5  
pandas==2.0.3  
seaborn==0.12.2   
torch==2.2.2  
tqdm==4.66.5  
transformers==4.32.1 

Packages: `builtins`, `sys`, `matplotlib.pyplot`,
 `random`, `torch.nn`, `pkg_resources`, `types`
## Contact 📩
For any questions or inquiries, feel free to reach out:
- **Email:** sevilaymunire68@gmail.com
- **LinkedIn:** [Sevilay Munire Girgin](www.linkedin.com/in/sevilay-munire-girgin-8902a7159)
Thank you for visiting my project repository. Happy and accurate classification! 💕
## License 🔐
The project is licensed under the MIT License.
