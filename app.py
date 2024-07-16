from flask import Flask, request, jsonify
from flask_cors import CORS
from textsum import summarize_text
from model import model
import torch

app = Flask(__name__)
CORS(app)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json(force=True)
    text = data.get('text', '')
    use_gpu = request.args.get('use_gpu', 'true').lower() == 'true'

    if not text:
        return jsonify({"error": "No text provided"}), 400

    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    summary = summarize_text(text)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
