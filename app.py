from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

API_URL = "https://api-inference.huggingface.co/pipeline/text-classification/Sinanmz/Movie_Genre_Classifier"
HF_TOKEN = os.environ.get("HF_TOKEN")

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    user_text = data.get('text', '')
    
    if not user_text:
        return jsonify({'error': 'No text provided'}), 400

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": user_text}

    try:
        # Use requests directly. It handles standard container DNS mapping automatically.
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()[0]
            best_guess = max(result, key=lambda x: x['score'])
            
            return jsonify({
                'genre': best_guess['label'],
                'confidence': round(best_guess['score'] * 100)
            })
        else:
            return jsonify({'error': f"AI platform error code: {response.status_code}"}), response.status_code

    except Exception as e:
        return jsonify({'error': f"Outbound network resolution error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)