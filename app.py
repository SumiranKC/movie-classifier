from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# 1. The correct, updated endpoint for the new router architecture
API_URL = "https://router.huggingface.co/hf-inference/v1/chat/completions"
HF_TOKEN = os.environ.get("HF_TOKEN")

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    user_text = data.get('text', '')
    
    if not user_text:
        return jsonify({'error': 'No text provided'}), 400

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # 2. REQUIRED SCHEMA CHANGE: The new router demands the messages format
    payload = {
        "model": "Sinanmz/Movie_Genre_Classifier",
        "messages": [
            {
                "role": "system", 
                "content": "You are a movie classification engine. Analyze the movie plot and respond with ONLY the predicted genre name and nothing else (e.g., Sci-Fi, Drama, Action)."
            },
            {
                "role": "user", 
                "content": user_text
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            # Extract the raw text string response from the new chat completions structure
            predicted_genre = result['choices'][0]['message']['content'].strip()
            
            # Keeps the exact object properties your index.html file expects to see
            return jsonify({
                'genre': predicted_genre,
                'confidence': '100'  # Hardcoded text because the new router text API doesn't pass individual label float scores anymore
            })
        else:
            return jsonify({'error': f"HuggingFace Router rejected request: {response.text}"}), response.status_code

    except Exception as e:
        return jsonify({'error': f"Cloud Routing Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)