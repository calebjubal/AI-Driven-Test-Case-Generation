from transformers import pipeline

# Load the fine-tuned model
model_path = './results'
nlp_pipeline = pipeline('text-classification', model=model_path)

# Example requirement
requirement = "The system shall refresh the display every 60 seconds."

# Generate a test case
test_case = nlp_pipeline(requirement)
print(test_case)
