from pyngrok import ngrok
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import numpy as np
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Load model
summarizer = pipeline('summarization')

# Start flask app and set to ngrok
app = Flask(__name__)

public_url = ngrok.connect()
print(' * ngrok tunnel:', public_url)

@app.route('/')
def initial():
    return render_template('index.html')

@app.route('/submit-text-abs', methods=['POST'])
def generate_summary_abs():
    prompt = request.form['summarizationInput']
    print(f"Generating summary of: {prompt}")

    summary = summarizer(prompt)
    print("Summary generated!")

    #return jsonify({'abstractiveOutput': summary[0]['summary_text']})
    return summary[0]['summary_text']


@app.route('/submit-text-ext', methods=['POST'])
def generate_summary_ext():
  num_sentences=3
  text = request.form['summarizationInput']

  sentences = sent_tokenize(text)

  tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
  preprocessed_sentences = [" ".join(words) for words in tokenized_sentences]
  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
  similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

  scores = np.sum(similarity_matrix, axis=1)
  ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
  top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])

  summary = ' '.join([sentences[idx] for score, idx in top_sentences])

  return summary

if __name__ == '__main__':
    app.run(port=80)
