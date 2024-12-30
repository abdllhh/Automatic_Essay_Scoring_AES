# Automatic_Essay_Scoring_AES
Automatic essay scoring by extracting multiple textual features, and through just essays and target scores

This project implements an Automatic Essay Scoring (AES) system using Machine Learning (ML) and Natural Language Processing (NLP) techniques. The system is designed to evaluate and score essays based on their linguistic and semantic features. The primary focus was to develop models leveraging deep learning methods such as Artificial Neural Networks (ANNs) and Long Short-Term Memory (LSTM) networks. 

The dataset consists of 12,977 essays from the Hewlett Foundation's Automated Essay Scoring (AES) dataset on Kaggle. Each essay includes:
Text: The raw essay content.
Score: The quality score assigned to the essay

#### dataset after feature extraction uploaded to Huggingface, can be accessed from: https://huggingface.co/datasets/abdlh/Dataset_Automatic_Essay_Scoring_Essay-EssayScore_and_24_textual_features/blob/main/Updated_Processed_Data.csv ####

## Preprocessing ##
#### as done in the 'new_dataset_from_old_dataset.ipynb' file ####

Data Cleaning:
Removed special characters, redundant spaces, and non-alphanumeric content.
Applied case normalization (converted text to lowercase).

Tokenization:
Split essays into words and sentences using NLTK.
Stopword Removal and Lemmatization:
Removed common stopwords.
Used SpaCy for lemmatization.

Feature Extraction: Extracted 20+ key features such as:
Linguistic Features: Word count, sentence count, average word length.
Readability Scores: Flesch Reading Ease, Gunning Fog Index.
Semantic Features: Sentiment analysis scores using TextBlob.
POS Tagging: Distribution of parts of speech.
Other Custom Features: Clause density, punctuation counts, argumentative tone, and more.

Embeddings:
Generated semantic embeddings using SpaCy's pre-trained language model for essay vectorization.

## Models and Training ##
#### as in 'using_models.ipynb' and 'new_dataset_from_old_dataset.ipynb' ####
Models:
Linear Regression:
Used as a baseline model with the extracted features.

Artificial Neural Network (ANN):
Input Layer: Matches the feature size.
Hidden Layers: Two layers (64 and 32 nodes) with ReLU activation.
Output Layer: Single regression neuron with linear activation.

Long Short-Term Memory (LSTM):
Input: Sequential embeddings for time-series analysis.
Architecture: One LSTM layer (64 units) followed by dense layers.

Recurrent Neural Network (RNN):
Similar to the LSTM architecture but using a SimpleRNN layer.

## Evaluation ##
Metrics:
MSE (Mean Squared Error)
MAE (Mean Absolute Error)
QWK (Quadratic Weighted Kappa)

Findings:
Models were evaluated on an 80%-20% train-test split.
### A baseline LSTM (as trained in 'lstm_file.py') trained directly on essay text with fewer features outperformed others but violated the 20+ features constraint. ###
Rest of the models trained on all 20+ features did not perform well on poor essays, though all gave great essays higher marks 

