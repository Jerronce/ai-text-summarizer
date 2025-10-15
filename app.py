from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import nltk
from bs4 import BeautifulSoup
import requests
import PyPDF2
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Initialize summarization pipeline
try:
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
except Exception as e:
    print(f'Error loading model: {e}')
    summarizer = None

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': summarizer is not None
    })

@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        text = data.get('text', '')
        max_length = data.get('max_length', 130)
        min_length = data.get('min_length', 30)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not summarizer:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Limit input text length
        if len(text) > 10000:
            text = text[:10000]
        
        # Generate summary
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        
        return jsonify({
            'summary': summary[0]['summary_text'],
            'original_length': len(text),
            'summary_length': len(summary[0]['summary_text'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize-url', methods=['POST'])
def summarize_url():
    try:
        data = request.json
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Fetch webpage content
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        if not text:
            return jsonify({'error': 'No text found in URL'}), 400
        
        # Limit text length
        if len(text) > 10000:
            text = text[:10000]
        
        # Generate summary
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        
        return jsonify({
            'summary': summary[0]['summary_text'],
            'url': url,
            'original_length': len(text)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract-keywords', methods=['POST'])
def extract_keywords():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Simple keyword extraction using NLTK
        tokens = nltk.word_tokenize(text.lower())
        from nltk.corpus import stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
        
        # Filter out stopwords and short words
        keywords = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 3]
        
        # Get most common keywords
        from collections import Counter
        keyword_freq = Counter(keywords)
        top_keywords = [word for word, freq in keyword_freq.most_common(10)]
        
        return jsonify({'keywords': top_keywords})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('Starting AI Text Summarizer API...')
    print('Model:', 'facebook/bart-large-cnn')
    app.run(host='0.0.0.0', port=5000, debug=True)
