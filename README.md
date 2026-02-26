# 📰 Fake News Detection Project

This project was developed as part of the **Elevate Labs Internship Program**.

## 📌 Objective
To classify news articles as **Fake** or **Real** using Natural Language Processing (NLP) and Machine Learning techniques.

---

## 🛠 Tools & Technologies
- Python  
- Pandas, NumPy  
- Scikit-learn  
- NLTK  
- TF-IDF Vectorizer  
- Logistic Regression  
- Jupyter Notebook  
- Streamlit

---
## 📂 Project Files
- `fake_news_classifier.ipynb` – Model training notebook  
- `app.py` – Streamlit web application  
- `Fake_News_Detection_Report.pdf` – Project report
  
---
## 📁 Dataset

This project uses the **Fake and Real News Dataset** from Kaggle.

The dataset contains:
- `Fake.csv` – Fake news articles  
- `True.csv` – Real news articles  

Dataset link:  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset  

Due to GitHub file size limits, the dataset files are not included in this repository.  
Please download the dataset from Kaggle and place the files inside a folder named `data/` as shown below:

Fake-News-Detection  
│  
└── data  
    ├── Fake.csv  
    └── True.csv  

---
## 💾 Model File

The trained model file (`model_bundle.pkl`) is not included in this repository because its size exceeds GitHub’s 25MB file upload limit.

You can generate the model file by running the notebook:

fake_news_classifier.ipynb

This will train the model and save `model_bundle.pkl` locally for use with the Streamlit application.

---

## ⚙ Steps Involved
1. Collected Fake and Real news dataset from Kaggle  
2. Cleaned and preprocessed text data  
3. Applied TF-IDF vectorization  
4. Trained Logistic Regression model  
5. Evaluated using accuracy, precision, recall, and F1-score  
6. Built a Streamlit web application for predictions  

---

## 📊 Results
- Accuracy: ~99%  
- Precision: 0.99  
- Recall: 0.99  
- F1-score: 0.99  

---

## ▶ How to Run the Project

**1️⃣ Install Required Libraries**

Make sure Python is installed, then run:
```bash
pip install -r requirements.txt
```

**2️⃣ Train the Model (Run the Notebook)**

Open and run the Jupyter Notebook to train the model and generate the model file:
```bash
fake_news_classifier.ipynb
```
This will create the trained model file:
```bash
model_bundle.pkl
```
**3️⃣ Run the Streamlit Web App**

After the model file is generated, run the Streamlit app using:
```bash
python -m streamlit run app.py
```
**4️⃣ Open in Browser**

Streamlit will open automatically in your browser at:
```bash
http://localhost:8501
```
----
## 🎯 Conclusion

This project demonstrates how Machine Learning and NLP can be applied to detect fake news effectively.
It provides a simple yet powerful web application for users to check the authenticity of news articles.
The project also highlights the importance of data preprocessing and model evaluation in real-world applications.
