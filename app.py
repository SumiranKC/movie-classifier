from flask import Flask, request, jsonify
from flask_cors import CORS
import urllib.request
import json
import os

app = Flask(__name__)
CORS(app)  # This allows your HTML file to communicate with this backend server

API_URL = "https://api-inference.huggingface.co/models/Sinanmz/Movie_Genre_Classifier"
HF_TOKEN = os.getenv("HF_TOKEN")

@app.route('/classify', methods=['POST'])
def classify():
    # 1. Grab the movie text sent from the HTML webpage
    data = request.json
    user_text = data.get('text', '')
    
    if not user_text:
        return jsonify({'error': 'No text provided'}), 400

    # 2. Package the data and password for the Hugging Face API
    payload = json.dumps({"inputs": user_text}).encode('utf-8')
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        # 3. Fire the request directly to the AI server using standard Python tools
        req = urllib.request.Request(API_URL, data=payload, headers=headers, method='POST')
        
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode('utf-8'))[0]
            
            # 4. Find the genre with the highest confidence calculation
            best_guess = max(result, key=lambda x: x['score'])
            
            # 5. Send clean numbers back to your HTML webpage
            return jsonify({
                'genre': best_guess['label'],
                'confidence': round(best_guess['score'] * 100)
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Starts the local server on your computer
    # app.run(port=5000)
    # Tells the cloud server to listen to all incoming web traffic
    app.run(host='0.0.0.0', port=5000)