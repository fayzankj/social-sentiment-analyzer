# Sentiment Analysis on Social Media Posts
This project aims to develop a robust sentiment analysis tool that classifies the emotional tone of social media posts, specifically tweets, as either **positive** or **negative**. By leveraging the **Sentiment 140** dataset, this tool provides valuable insights into public sentiment, trends, and opinions expressed across Twitter.

## 🚀 Project Highlights

### **Data Collection:**
- Seamlessly integrates the **Kaggle API** to download the **Sentiment 140 dataset**, which contains over **1.6 million labeled tweets**.
  
### **Data Preprocessing:**
- Cleans and prepares the raw text data by performing the following:
  - **Converting text to lowercase** for uniformity.
  - **Removing non-alphabetic characters** to retain meaningful content.
  - **Tokenizing and stemming** using **NLTK's PorterStemmer** to extract core meanings.
  - **Filtering out stopwords**, reducing noise and enhancing model efficiency.

### **Machine Learning Model:**
- Converts preprocessed text into **numerical vectors** using **TF-IDF vectorization**, capturing important keywords and context.
- Employs **Logistic Regression** for sentiment classification, distinguishing between **positive** and **negative** sentiments in tweets.

### **Model Evaluation:**
- Evaluates model performance using **accuracy metrics** on both training and testing datasets.
- Visualizes performance comparison between training and testing accuracy, offering intuitive insights into the model’s effectiveness.

### **Persistent Model:**
- Saves the trained model using **pickle**, making it easy for future use or deployment without needing to retrain.

### **Dataset Management:**
- Automates dataset download, extraction, and preparation processes, ensuring a smooth workflow and reducing manual intervention.
- Handles **missing values** and **renames columns** for consistency across datasets.

## 📊 Dataset Details

The **Sentiment 140 dataset** contains **1.6 million tweets** that are pre-labeled as either:

- **0** (Negative sentiment)
- **1** (Positive sentiment)

You can access the dataset [here](https://www.kaggle.com/datasets/kazanova/sentiment140).

## 📈 Performance Metrics

- **Training Accuracy:** 81%
- **Testing Accuracy:** 78%

These results demonstrate the model's ability to generalize well to unseen data, with solid performance on both training and testing datasets.

## 🛠️ Technologies and Libraries

This project utilizes a range of powerful Python libraries to process data, build the model, and evaluate performance:

- **Python**: The core programming language used for the project.
- **Pandas**: For efficient data manipulation and analysis.
- **NLTK**: For natural language processing, including tokenization, stemming, and stopword removal.
- **Scikit-learn**: For implementing machine learning algorithms like Logistic Regression and TF-IDF vectorization.
- **Matplotlib**: For visualizing model performance and results.
- **Pickle**: For saving and loading the trained model.

## 📝 Getting Started

### 1. Clone this repository:

```bash
git clone https://github.com/yourusername/sentiment-analysis-social-media.git
```
### 2. Install the required dependencies:

```bash
pip install -r requirements.txt
```
### 3. Set up your Kaggle API token:
- Download your kaggle.json API token from Kaggle.
- Place the kaggle.json file in the project folder to enable dataset downloading.

### 4. Model Usage  
The trained Logistic Regression model is saved and ready for use. You can easily load the model and classify the sentiment of new tweets using the predict_sentiment function.
```bash
sentiment = predict_sentiment("I love this new product!")
print("Sentiment:", "Positive" if sentiment == 1 else "Negative")
```

## 💼 Contributing
Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests. Whether it’s improving data preprocessing, experimenting with new models, or refining the user interface, your contributions are highly valued!
