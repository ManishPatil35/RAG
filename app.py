from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load tokenizer and model for text generation
model_name = 't5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load text generation pipeline
gen_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def split_into_chunks(text, chunk_size=200):
    """Splits the given text into chunks of specified size."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def retrieve_relevant_chunks(question, context_chunks, top_n=3):
    """Retrieves the top_n most relevant chunks from the context based on the question."""
    vectorizer = TfidfVectorizer().fit_transform([question] + context_chunks)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    relevant_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return [context_chunks[i] for i in relevant_indices]

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        if not request.is_json:
            logging.warning("Request must be JSON.")
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.json
        question = data.get('question')

        if not question:
            logging.warning("No question provided.")
            return jsonify({"error": "No question provided"}), 400

        # Read context from luke_skywalker.txt
        file_path = 'luke_skywalker.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                context = file.read()
        else:
            logging.warning("Context file not found.")
            return jsonify({"error": "Context file not found"}), 400

        # Split context into chunks
        context_chunks = split_into_chunks(context)

        # Retrieve the top 3 relevant chunks
        relevant_chunks = retrieve_relevant_chunks(question, context_chunks, top_n=3)
        combined_context = ' '.join(relevant_chunks)

        # Generate detailed response using the text generation model
        gen_input = f"question: {question} context: {combined_context}"
        gen_response = gen_pipeline(gen_input, max_length=150, do_sample=False)
        detailed_answer = gen_response[0]['generated_text'].strip()

        return jsonify({'answer': detailed_answer})

    except Exception as e:
        logging.error(f"Error in /ask endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
