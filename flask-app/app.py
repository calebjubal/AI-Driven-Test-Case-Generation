from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the fine-tuned model
model_path = './results'
nlp_pipeline = pipeline('text-classification', model=model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_test_case', methods=['POST'])
def generate_test_case():
    data = request.json
    requirement = data['requirement']
    test_case = nlp_pipeline(requirement)
    return jsonify(test_case)

if __name__ == '__main__':
    app.run(debug=True)
