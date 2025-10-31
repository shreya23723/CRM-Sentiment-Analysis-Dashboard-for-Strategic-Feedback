1. Project Overview
This project implements sentiment analysis on customer reviews using Natural Language Processing (NLP) and machine learning models. Initially, we applied Naïve Bayes for classification but later transitioned to DistilBERT, a transformer-based model, to improve accuracy. The results are then visualized using Power BI to provide meaningful insights into customer sentiments.

This system helps businesses analyze large volumes of reviews, identify sentiment trends, and make data-driven decisions to improve products and services.

2. Features
✅ Preprocessing of Customer Reviews – Tokenization, stopword removal, and text normalization.
✅ Sentiment Classification – Classifies reviews into Positive or Negative using DistilBERT.
✅ Batch Processing for Large Datasets – Ensures efficient handling of large review datasets.
✅ Power BI Dashboard – Provides interactive visualizations for sentiment trends and insights.
✅ Performance Comparison – Evaluates Naïve Bayes vs. DistilBERT on multiple metrics.

3. Technologies Used
Programming & AI/ML Tools
Python 3.x – Core programming language
Transformers (Hugging Face) – DistilBERT model for sentiment analysis
NLTK (Natural Language Toolkit) – Text preprocessing (stopwords, tokenization)
Pandas & NumPy – Data handling and processing
Matplotlib & Seaborn – Data visualization (for initial analysis)
Scikit-learn – Traditional ML model (Naïve Bayes)
Visualization & Deployment
Power BI – Data visualization and interactive dashboard
Jupyter Notebook / VS Code – Development environment
Google Colab (Optional) – Cloud-based execution

4. Dataset Details
📌 Dataset Used: Amazon Product Reviews Dataset
📌 Key Columns in Dataset:
reviewText – The main customer review text.
overall – Star rating (1-5).
helpful – Upvotes on a review.
reviewTime – Date of the review.
sentiment – (Generated column: Positive or Negative).

5. Project Workflow
Step 1: Data Preprocessing
Load the dataset (amazon_review.csv).
Drop missing values and convert text to lowercase.
Remove stopwords and tokenize sentences.
Split long reviews into smaller chunks to prevent model errors (max length = 512).

Step 2: Sentiment Analysis Using NLP
Baseline Model: Naïve Bayes (Traditional ML).
Vectorized text using TF-IDF.
Achieved 78.6% accuracy.
Final Model: DistilBERT (Deep Learning).
Used pre-trained DistilBERT model for classification.
Achieved 92.3% accuracy.
Batch Processing: Handled large datasets efficiently.

Step 3: Performance Evaluation
Metric	Naïve Bayes 	DistilBERT
Accuracy	78.6%	        92.3%
Precision	76.2%	        91.5%
Recall	    74.8%	        92.8%
F1-Score	75.5%	        92.1%
Training Time	10 sec	4 min (GPU)
Inference Time	0.02 sec	0.1 sec
Step 4: Power BI Visualization
The final sentiment analysis results are stored in CSV format and visualized in Power BI.

Key Visuals in the Dashboard:
✅ Sentiment Distribution – Count of Positive & Negative Reviews
✅ Review Sentiment Trends Over Time – Understanding sentiment shifts
✅ Helpful Votes by Sentiment – Analyzing which reviews get more engagement
✅ Star Rating vs. Sentiment – Correlation between user ratings and model predictions

6. Installation & Setup
A. Environment Setup
1️⃣ Clone the repository (if using GitHub):


git clone https://github.com/your-repo/sentiment-analysis.git
cd sentiment-analysis
2️⃣ Create a Virtual Environment (Recommended):
python -m venv .venv
source .venv/bin/activate  # For Mac/Linux
.venv\Scripts\activate      # For Windows
B. Install Dependencies

pip install -r requirements.txt
C. Run Sentiment Analysis Script

python sentiment_analysis.py
D. Load the Results into Power BI
Open Power BI.
Import sentiment_analysis_results.csv.
Design the dashboard using the provided visualizations.
7. Challenges Faced & Solutions
Challenge 1: Handling Large Reviews (Text Length > 512 Tokens)
🔹 Issue: Transformer models have a token limit (512 words).
✅ Solution: We split long reviews into multiple chunks, ensuring all text is processed correctly.

Challenge 2: Computational Constraints
🔹 Issue: Transformer models require high processing power.
✅ Solution: Used batch processing and GPU acceleration to improve efficiency.

Challenge 3: Ensuring High Accuracy & Context Understanding
🔹 Issue: Traditional ML models like Naïve Bayes struggled with negations and sarcasm.
✅ Solution: Transitioned to DistilBERT, which captures context better using self-attention mechanisms.

8. Real-World Applications
🚀 E-commerce – Analyze customer feedback for better product recommendations.
🚀 Customer Support – Automate sentiment detection for service improvements.
🚀 Finance – Track market sentiment through customer and investor feedback.
🚀 Healthcare – Assess patient reviews for service enhancements.
🚀 Social Media Monitoring – Detect negative sentiment trends in brand mentions.

9. Future Enhancements
🔹 Multilingual Sentiment Analysis – Extend support to multiple languages.
🔹 Aspect-Based Sentiment Analysis – Identify sentiment for specific product features.
🔹 Real-Time Sentiment Tracking – Process live reviews and feedback streams.

10. Contributors
👨‍💻 Author: Shreya Santosh Bartakke
📩 Contact: shreyab2307@gmail.com
📂 GitHub Repo: https://github.com/shreya23723/CRM-Sentiment-Analysis-Dashboard-for-Strategic-Feedback

11. License
📄 MIT License – Open-source for educational and research purposes.

12. Final Thoughts
This project successfully demonstrates how NLP and deep learning can enhance sentiment analysis accuracy. By integrating DistilBERT for classification and Power BI for visualization, this system offers a scalable, high-performance sentiment analysis solution for various industries.
