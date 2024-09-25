# Fake News Detection Using Logistic Regression

## Project Overview

This project focuses on building a machine learning model to detect fake news articles. The model is trained using a dataset of news articles and utilizes Logistic Regression for classification. The primary goal is to classify news articles as either "real" or "fake" based on their text features such as title and author.

#### Features

- **Data Pre-processing**: 
  - Removal of *stopwords* (e.g., 'I', 'the', 'it') which are commonly used but provide little value for distinguishing real vs. fake news.
  - Handling missing values by replacing *Null* entries with an empty string.
  - Merging the `title` and `author` columns into a new feature for more comprehensive text-based analysis.
  
- **Text Processing**:
  - Tokenization and stemming are performed using the **Porter Stemmer**.
  - The model uses **TF-IDF Vectorization** to convert the textual data into numerical form, enabling machine learning models to process it.

#### Libraries Used

- **NumPy**
- **Pandas**
- **NLTK** (for stopwords and stemming)
- **Scikit-learn** (for Logistic Regression, TF-IDF Vectorizer, train/test split, and accuracy metrics)

#### Dataset(https://www.kaggle.com/c/fake-news/data?select=train.csv)

The dataset consists of news articles, including attributes such as the title and author. It is loaded from a CSV file and preprocessed to remove missing values and merge relevant features.

#### Model

The model used in this project is **Logistic Regression**, which is suitable for binary classification tasks like fake news detection. The model is trained on preprocessed text data and evaluated using accuracy as the performance metric.

