from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

from transformers import pipeline
summarizer = pipeline('summarization')

# Load model
pipe = pipeline('summarization')
pipe.to("cuda")

# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)


@app.route('/')
def initial():
    return render_template('index.html')


@app.route('/submit-text', methods=['POST'])
def generate_image():
    prompt = request.form['abstractiveInput']
    print(f"Generating an image of {prompt}")

    summary = pipe(prompt)
    print("Summary generated!")


    print("Sending summary ...")
    return render_template('index.html', abstractiveOutput=summary)


if __name__ == '__main__':
    app.run()

