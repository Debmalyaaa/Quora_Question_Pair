# 🤖 Duplicate Question Detection using Quora Question Pairs

This project focuses on detecting **duplicate questions** using natural language processing (NLP) and machine learning techniques. Based on the [Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs/data) dataset, the goal is to classify whether two questions are semantically similar or not.

---

## 📂 Dataset

- **Source**: [Kaggle - Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs/data)
- **File**: `train.csv`
- **Attributes**:
  - `qid1`, `qid2`: Unique question IDs
  - `question1`, `question2`: The actual questions
  - `is_duplicate`: Target label (1 = duplicate, 0 = not duplicate)

---

## 🧰 Technologies Used

- **Python**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn** – traditional ML models and metrics
- **XGBoost**
- **TensorFlow / Keras** – for RNN and LSTM
- **Gensim** – Word2Vec embeddings
- **NLTK**, **FuzzyWuzzy**, **Distance** – text processing
- **Streamlit** – for web deployment

---

## 📊 Key Features

### ✅ Data Preprocessing
- Lowercasing, punctuation removal, and HTML tag stripping
- Contraction expansion and token filtering
- Word embeddings using Word2Vec

### ✅ Feature Engineering
- 21 handcrafted features using:
  - Token-based and fuzzy string similarity
  - Common word/token counts
  - Length ratios and substring overlaps

### ✅ Models Implemented
- `RandomForestClassifier`
- `XGBoost Classifier`
- `Support Vector Machine (SVC)`
- `Recurrent Neural Network (RNN)`
- `Long Short-Term Memory (LSTM)`

### ✅ Performance Evaluation
- Accuracy, F1-Score
- Confusion Matrix
- Model Comparison Plots

---

## 🌐 Web App

### 🚀 Deployment
Built using **Streamlit**. You can input two questions and get a prediction (Duplicate / Not Duplicate).

### 📦 Files Used
- `app.py`: Main app script
- `model.pkl`: Trained classifier (e.g., Random Forest)
- `word2vec_model.pkl`: Pre-trained Gensim Word2Vec model

---

## 📈 Evaluation Metrics

| Metric         | Description                            |
|----------------|----------------------------------------|
| Accuracy        | Overall correct predictions            |
| F1 Score        | Balance between precision & recall     |
| Confusion Matrix| Breakdown of TP, FP, FN, TN            |
| Log Loss        | Used with probabilistic classifiers    |


---

## 🙏 Acknowledgments

- [Kaggle Quora Dataset](https://www.kaggle.com/competitions/quora-question-pairs/data)
- TensorFlow, Scikit-learn, Gensim, NLTK, FuzzyWuzzy
- All open-source contributors and GitHub repositories that inspired this work
