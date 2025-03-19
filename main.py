import torch
from gliner import GLiNER

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained NERCat model and tokenizer
model = GLiNER.from_pretrained("ugiat/NERCat").to(device)

# Input text for Named Entity Recognition (NER)
text = "La Universitat de Barcelona és una de les institucions educatives més importants de Catalunya."

# Labels representing the possible entity types
labels = [
    "Person",
    "Facility",
    "Organization",
    "Location",
    "Product",
    "Event",
    "Date",
    "Law"
]

# Predict entities with a confidence threshold of 0.5
entities = model.predict_entities(text, labels, threshold=0.5)

# Display recognized entities and their labels
for entity in entities:
    print(entity["text"], "=>", entity["label"])
