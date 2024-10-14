from flask import Flask, request, jsonify, render_template
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from translation_model import preprocess_text, post_process_translations, build_transformer_model, translate_text_mbart

app = Flask(__name__)

# Load the mBART model and tokenizer once when the server starts
model, tokenizer = build_transformer_model()

# Route to serve the homepage (the UI)
@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html is in the templates folder

# API endpoint for translating text
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    input_text = data['text']
    direction = data['direction']  # '1' for Marathi to English, '2' for English to Marathi

    # Determine source and target languages based on the direction
    if direction == '1':
        src_lang = 'mr_IN'
        tgt_lang = 'en_XX'
    else:
        src_lang = 'en_XX'
        tgt_lang = 'mr_IN'

    # Call the translation function from your model
    translated_text = translate_text_mbart(input_text, model, tokenizer, src_lang, tgt_lang)

    # Return the translated text as a JSON response
    return jsonify({'translated_text': translated_text})

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Change port to 8000 or any other available port
