from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Input text
text = "The movie was incredibly boring and I couldn't finish it."

# Candidate labels for binary classification
labels = ["positive", "negative"]

# Perform classification
result = classifier(text, labels)

# Display results
print(f"Text: {text}")
print(f"Predicted label: {result['labels'][0]}")
print(f"Confidence score: {result['scores'][0]}")
