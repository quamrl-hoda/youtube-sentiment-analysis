# app.py

# First, set matplotlib backend BEFORE any other imports that might import pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

# Now import other modules
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
import matplotlib.dates as mdates
from dotenv import load_dotenv
import os
import sys
import nltk
from mlflow.tracking import MlflowClient

# Load environment variables first
load_dotenv()

# NLTK Setup
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded
# We use quiet=True to avoid spamming logs, but allow it to download if missing.
# We explicitly download 'omw-1.4' which is required for WordNet in newer NLTK versions.
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# FORCE NLTK resources to load eagerly (non-lazy)
# This fixes the "LazyCorpusLoader" threading error in Flask
try:
    # Accessing the corpus forces it to load
    wn.ensure_loaded()
    stopwords.ensure_loaded()
    
    # Initialize global instances
    STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    LEMMATIZER = WordNetLemmatizer()
    
    # Test them immediately to fail fast if something is wrong
    LEMMATIZER.lemmatize('test')
    print("✅ NLTK Resources loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL NLTK ERROR: {e}")
    # In production we might want to exit, but for now let's just print
    # sys.exit(1)

# Get DAGSHUB_TOKEN from environment variable
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '').strip()
if not DAGSHUB_TOKEN:
    print("WARNING: DAGSHUB_TOKEN not found in environment variables")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords using pre-loaded set
        comment = ' '.join([word for word in comment.split() if word not in STOP_WORDS])

        # Lemmatize the words using pre-loaded lemmatizer
        comment = ' '.join([LEMMATIZER.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Set MLflow tracking credentials
if DAGSHUB_TOKEN:
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'quamrl-hoda'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
else:
    print("WARNING: MLflow tracking may not work without DAGSHUB_TOKEN")

def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    """Load model and vectorizer from MLflow and local storage."""
    try:
        # Set the tracking URI to your DagsHub repository
        mlflow.set_tracking_uri('https://dagshub.com/quamrl-hoda/youtube-sentiment-analysis.mlflow')
        
        # Load model from MLflow registry
        client = MlflowClient()
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # More robust path handling
        vectorizer_path = os.path.join(os.path.dirname(__file__), 'models', 'tfidf_vectorizer.pkl')
        if not os.path.exists(vectorizer_path):
            vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
        if not os.path.exists(vectorizer_path):
            vectorizer_path = os.path.join('artifacts', 'models', 'tfidf_vectorizer.pkl')
        vectorizer = joblib.load(vectorizer_path)
        print(f"✅ Model and vectorizer loaded successfully from {vectorizer_path}")
        
        return model, vectorizer
    except Exception as e:
        print(f"❌ Error loading model and vectorizer: {e}")
        raise e

# Initialize the model and vectorizer
try:
    model, vectorizer = load_model_and_vectorizer(
        "youtube_sentiment_lgbm", 
        "Staging", 
        "models/tfidf_vectorizer.pkl"  # Update path as needed
    )
except Exception as e:
    print(f"CRITICAL: Failed to load model: {e}")
    model, vectorizer = None, None

@app.route('/')
def home():
    return jsonify({
        "message": "YouTube Sentiment Analysis API",
        "status": "running",
        "endpoints": {
            "/": "This message",
            "/predict": "POST - Predict sentiment for comments"
        }
    })
@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 503
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert to DataFrame with feature names to match model requirements
        # Check for sklearn version compatibility
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()
            
        transformed_comments_df = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

        # Make predictions
        predictions = model.predict(transformed_comments_df).tolist()
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 503
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
       # Preprocess each comment before vectorizing
      preprocessed_comments = [preprocess_comment(comment) for comment in comments]
      # Transform comments using the vectorizer
      transformed_comments = vectorizer.transform(preprocessed_comments)

      # Convert to DataFrame with feature names to match model requirements
      # Check for sklearn version compatibility
      if hasattr(vectorizer, 'get_feature_names_out'):
          feature_names = vectorizer.get_feature_names_out()
      else:
          feature_names = vectorizer.get_feature_names()
          
      transformed_comments_df = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)

      # Make predictions
      predictions = model.predict(transformed_comments_df).tolist()
        
      # Convert predictions to strings for consistency
      predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": category} for comment, category in zip(comments, predictions)]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    # Run the app
    print("🚀 Starting Flask API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)