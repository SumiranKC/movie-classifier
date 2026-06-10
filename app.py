from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Keeps communication open with your GitHub Pages frontend

# 1. FIXED: Correct global OpenAI-compatible base path for the new router
API_URL = "https://router.huggingface.co/v1/chat/completions"
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
    
    # 2. FIXED: Swapped to a globally whitelisted model ID on the router network
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "system", 
                "content": "You are a movie classification engine. Analyze the movie plot and respond with ONLY the single best predicted genre name and nothing else (e.g., Sci-Fi, Drama, Action)."
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
            # Extract the response string directly from the messaging array
            predicted_genre = result['choices'][0]['message']['content'].strip()
            
            # Returns data in the exact format your index.html file needs
            return jsonify({
                'genre': predicted_genre,
                'confidence': 'Verified'
            })
        else:
            return jsonify({'error': f"HuggingFace Router rejected request: {response.text}"}), response.status_code

    except Exception as e:
        return jsonify({'error': f"Cloud Routing Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)