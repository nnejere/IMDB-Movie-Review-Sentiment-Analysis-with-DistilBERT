# 🎬 Sentiment Analysis on IMDb Reviews with DistilBERT

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)  
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers/index)  
[![Deep Learning](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)  

---

## 📊 Dataset Information  
- **Source:** IMDb Movie Reviews Dataset (100,000 samples)  
- **Description:** Contains **binary sentiment labels** (positive/negative) for movie reviews. Dataset is clean, balanced, and well-structured with three splits:  
  - **train:** 25,000 labeled reviews  
  - **test:** 25,000 labeled reviews  
  - **unsupervised:** 50,000 unlabeled reviews (optional for semi-supervised tasks)  
- **Features:**  
  - `text`: movie review content  
  - `label`: sentiment (0 = negative, 1 = positive)  

---

## 🎯 Problem Statement  
Understanding human emotions and opinions expressed in text is central to **Natural Language Processing (NLP)**. Sentiment analysis is critical for:  
- Customer feedback analysis  
- Social media monitoring  
- Product review interpretation  
- Automated moderation  

Traditional models rely on manual feature extraction, whereas **transformers like DistilBERT** learn contextual representations directly from text. This project fine-tunes DistilBERT on the IMDb dataset to classify reviews into positive or negative sentiment, balancing computational efficiency and performance.  

The model is deployed as a **Streamlit web application** and a **FastAPI endpoint** for real-time inference.  

---

## 📝 Objectives  
- Build a **binary sentiment classification system** for IMDb reviews.  
- Apply **transfer learning** and **fine-tuning** on DistilBERT for contextual understanding.  
- Evaluate performance using **accuracy, precision, recall, and F1-score**.  
- Deploy the model via **Streamlit** for interactive use and **FastAPI** for scalable inference.  
- Demonstrate an **end-to-end NLP pipeline**, from data exploration to deployment.  

---

## ⚙️ Methods and Approach  

### 🔹 Data Exploration & Preprocessing  
- Balanced dataset: 12,500 positive and 12,500 negative reviews per split.  
- Review lengths: 52–13,704 characters (avg ~1,325).  
- HTML tags removed from ~14,000 reviews.  
- Dataset clean, textually consistent, and ready for tokenization.  

### 🔹 Tokenization & Model Preparation  
- Tokenized with `DistilBertTokenizerFast`.  
- Maximum sequence length: 128–256 tokens.  
- PyTorch `DataLoaders` created for efficient batch processing.  

### 🔹 Model Architecture & Training  

#### Transfer Learning Phase (Phase 1)  
- **Trains only the classification head** for 2 epochs.  
- Base DistilBERT layers frozen to preserve pretrained knowledge.  
- **Results:**  
  - Training Loss decreased from 0.4741 → 0.4333  
  - Validation Accuracy rose from 80.68% → 80.94%  
- Validates pre-trained DistilBERT as an effective feature extractor.  

#### Fine-Tuning Phase (Phase 2)  
- Last 3 transformer layers (48 sub-layers) unfrozen; trained for 3 epochs.  
- GPU acceleration used (`cuda`).  
- **Results:**  
  - Training Loss decreased from 0.3518 → 0.2320  
  - Training Accuracy: 84.70% → 90.42%  
  - Validation Accuracy: 87.17% → 88.16%  
- Confirms optimal adaptation to IMDb review nuances.  

---

## 📈 Key Results  

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | 88.16% |
| **Precision** | 0.8864 |
| **Recall** | 0.8754 |
| **F1-Score** | 0.8809 |
| **Model** | DistilBERT (fine-tuned) |
| **Dataset** | IMDb Movie Reviews (Balanced) |

> Near-identical performance for negative (0) and positive (1) classes confirms **balanced and unbiased learning**.  

---

## 🚀 Recommendations  
- **Expand the Dataset:** Incorporate additional reviews from Rotten Tomatoes, Amazon, etc., to improve generalization.  
- **Experiment with Larger Models:** Explore BERT-base or RoBERTa for potentially higher contextual understanding (at higher computational cost).  
- **Cross-Validation:** Use k-fold CV for more reliable performance estimation.  
- **Hyperparameter Optimization:** Tune learning rate, batch size, and sequence length with frameworks like Optuna.  
- **Deployment Monitoring:** Integrate user feedback loops to flag misclassifications and continuously fine-tune in production.  
- **Explainability:** Use LIME or SHAP to interpret model decisions and improve trustworthiness.  

---

## 🏁 Conclusion  
This project demonstrates the power of **transfer learning and fine-tuning** using DistilBERT for sentiment analysis:  

- The model achieved **>80% validation accuracy** during transfer learning and improved to **88.16% test accuracy** after fine-tuning.  
- F1-Scores of **0.88 for both positive and negative classes** confirm robust, unbiased classification.  
- End-to-end pipeline includes **data exploration, preprocessing, tokenization, model training, evaluation, and deployment**.  
- The model is **production-ready**, capable of real-time inference via Streamlit and FastAPI.  
- With additional datasets, hyperparameter tuning, and explainability enhancements, the model can serve as a **foundation for more robust NLP applications** in customer feedback, brand monitoring, and social media sentiment tracking.  

---

Demo App: https://nnejere-ai-ml-inference-hub-app-298cc0.streamlit.app/
