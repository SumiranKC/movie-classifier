from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Keeps communication open with GitHub Pages frontend

# FIXED: Swapped out the decommissioned domain for the active serverless router endpoint
API_URL = "https://router.huggingface.co/hf-inference/models/Sinanmz/Movie_Genre_Classifier"
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
    payload = {"inputs": user_text}

    try:
        # Send the payload to the new Hugging Face router path
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()[0]
            # Find the genre label with the mathematical maximum confidence score
            best_guess = max(result, key=lambda x: x['score'])
            
            # Returns data in the exact format your index.html file expects
            return jsonify({
                'genre': best_guess['label'],
                'confidence': round(best_guess['score'] * 100)
            })
        else:
            return jsonify({'error': f"HuggingFace Router returned status code: {response.status_code}"}), response.status_code

    except Exception as e:
        return jsonify({'error': f"Cloud Routing Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Binds directly to the production cloud container port variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)