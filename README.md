


Comment Toxicity Detection
An end-to-end NLP and Deep Learning application for detecting potentially toxic comments using a PyTorch LSTM model.

Project Objective
The goal is to automatically identify potentially toxic comments so that large volumes of online content can be screened for moderation.

Key Features
Insights & Model Performance
Training/data statistics

Toxicity label distribution

Comment-length distribution

Accuracy, Precision, Recall, F1-score and ROC-AUC

Confusion matrix

LSTM vs RNN architecture comparison

Sample validation/test cases

Real-Time Prediction
Users can enter or paste a comment and receive:

Toxicity probability

Clean / potentially toxic classification

Configurable decision threshold

Cleaned/tokenized text used for inference

Bulk Prediction
Users can upload a CSV, select the comment-text column, score multiple comments, view the results, and download the predictions.

Machine Learning Workflow
Raw Comments
     |
     v
EDA / Data Exploration
     |
     v
Text Preprocessing
     |
     v
Tokenization / Numerical Representation
     |
     v
Sequence Padding
     |
     v
RNN and LSTM Training
     |
     v
Model Evaluation
     |
     v
Select Better Architecture
     |
     v
Save Trained Model
     |
     v
Streamlit Inference Application
     |
     +-------------------+
     |                   |
     v                   v
Real-Time Prediction   Bulk CSV Prediction
Model
Primary Model
LSTM (Long Short-Term Memory) implemented using PyTorch.

Comparison Model
A vanilla RNN is trained alongside the LSTM. The application compares the architectures using Accuracy, Precision, Recall, F1-score and ROC-AUC, with F1 used to select the better architecture for deployment.

Evaluation Metrics
Accuracy: Overall percentage of correct predictions.

Precision: How many predicted-toxic comments are actually toxic.

Recall: How many truly toxic comments are detected.

F1-score: Harmonic mean of precision and recall.

ROC-AUC: Measures classification discrimination across thresholds.

Confusion Matrix: Shows true positives, true negatives, false positives and false negatives.

Streamlit Application
The application provides three main tabs:

1. Insights & Model Performance
2. Real-Time Prediction
3. Bulk Prediction (CSV Upload)
Real-Time Prediction Flow
User Comment
    |
    v
Text Preprocessing
    |
    v
LSTM Inference
    |
    v
Toxicity Probability
    |
    v
Decision Threshold
    |
    +----> Potentially Toxic
    |
    +----> Clean
Bulk Prediction Flow
CSV Upload
    |
    v
Select Comment Column
    |
    v
LSTM Inference
    |
    v
Toxicity Score + Flag
    |
    v
Download Results CSV
Project Structure
A typical project structure is:

Comment_Toxicity/
│
├── app.py
├── lstm_pipeline.py
├── metrics.json
├── sample_cases.json
├── train_stats.json
├── sample_reviews.csv
│
└── notebook/
    └── toxicity_checkpoint.pth
Keep the checkpoint path in app.py consistent with the actual saved model location.

Important Deployment Principle
The Streamlit application does not train the model.

Training is performed offline. After training and evaluation, the model checkpoint is saved. The Streamlit application loads the saved checkpoint and performs inference on new comments.

TRAINING
    |
    +--> Train RNN / LSTM
    |
    +--> Evaluate
    |
    +--> Save checkpoint
                |
                v
DEPLOYMENT
    |
    +--> Load checkpoint
    |
    +--> Preprocess new comment
    |
    +--> Predict
    |
    +--> Display result
Installation
Create a virtual environment on Windows:

python -m venv venv
venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
If requirements.txt is not available:

pip install streamlit torch pandas numpy matplotlib seaborn
Run the Application
From the project root:

streamlit run app.py
CSV Input
The bulk prediction module accepts a CSV containing a comment-text column.

The current application can recognize common column names such as:

comment_text
text
comment
message
Example:

comment_text
"This is a great article."
"That was a terrible comment."
"I found this information useful."
Technology Stack
Python

PyTorch

LSTM / RNN

Natural Language Processing

Pandas

NumPy

Matplotlib

Seaborn

Streamlit

Project Story
This project follows an end-to-end machine learning workflow:

Understand the toxicity detection problem.

Explore the data and text characteristics.

Preprocess the comments for deep learning.

Train sequential models.

Compare RNN and LSTM.

Evaluate the models with multiple metrics.

Save the trained model.

Build the Streamlit inference application.

Support real-time and bulk predictions.

Limitations
The classifier can make mistakes with ambiguous language, sarcasm, context-dependent statements, or comments that differ substantially from the training data. Predictions should therefore be treated as a moderation aid rather than an absolute judgment.

Future Enhancements
Retrain using the complete training dataset.

Hyperparameter tuning.

Compare LSTM with Transformer models such as BERT.

Add multilingual toxicity detection.

Add model explainability.

Add inference monitoring and logging.

Containerize the application for production deployment.
