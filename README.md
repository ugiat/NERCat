
# <img src="https://openmoji.org/data/color/svg/1F3F4-E0065-E0073-E0063-E0074-E007F.svg" alt="Catalan Flag" width="40" height="40" style="vertical-align: middle;"> NERCat: Named Entity Recognition for Catalan Texts

NERCat is a fine-tuned Named Entity Recognition (NER) model for the Catalan language. It improves NER performance by leveraging a high-quality, expert-annotated dataset and addresses the scarcity of Catalan NER resources.

## Overview

- **Model:** Fine-tuned version of the GLiNER Knowledgator model (`knowledgator/gliner-bi-large-v1.0`)
- **Dataset:** Manually annotated Catalan-language television transcriptions with 13,732 named entity instances
- **Languages:** Catalan (ca) with occasional Spanish (es) code-switching
Classification
- **License:**  Apache-2.0 license


## 🛠️ NERCat Model
The NERCat model is based on the GLiNER Knowledgator architecture (`knowledgator/gliner-bi-large-v1.0`). It has been fine-tuned specifically for Named Entity Recognition in Catalan text, offering high accuracy in recognizing entities such as persons, locations, organizations, and more.

### Requirements 
To use NERCat, make sure you have the following dependencies installed:

- **Python:** 3.9 or higher
- **CUDA:** 11.7 or higher (if using GPU)

To install the required Python packages, use the following command:

```bash
pip install -r requirements.txt
```

### Usage
After installing the dependencies, load the `NERCat` model to perform inference on your input text. Below is an example of how to use it:

```python
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
```

### Expected Output
When running the example above, you should see output like the following:

```plaintext
Universitat de Barcelona => Organization
Catalunya => Location
```

## 📊 NERCat Dataset 

### Dataset Summary
The NERCat dataset consists of 9,242 sentences with 13,732 named entities across eight categories. The data was manually annotated to ensure high quality and consistency.

- **Person:** Names of individuals
- **Facility:** Physical infrastructures
- **Organization:** Companies, institutions, media
- **Location:** Geographical places
- **Product:** Commercial and cultural products
- **Event:** Named occurrences like festivals
- **Date:** Temporal expressions
- **Law:** Legal frameworks and judicial matters

### Example Data Instance

```json
{
    "tokenized_text": ["La", "Universitat", "de", "Barcelona", "és", "una", "de", "les", "institucions", "educatives", "més", "importants", "de", "Catalunya", "."],
    "ner": [
            [1, 3, "Organization"],
            [13, 13, "Location"]
    ]
}
```

## 📚 Citation
This project is based on the approach presented in the paper "NERCat: Fine-Tuning for Enhanced Named Entity Recognition in Catalan". You can read the full paper [here](https://github.com/ugiat/NERCat/blob/main/Catalan_GLiNER_Paper.pdf).

```
@misc{article_id,
  title        = {NERCat: Fine-Tuning for Enhanced Named Entity Recognition in Catalan},
  author       = {Guillem Cadevall Ferreres, Marc Bardeli Gámez, Marc Serrano Sanz, Pol Gerdt Basuillas, Francesc Tarres Ruiz, Raul Quijada Ferrero},
  year         = {2025},
  archivePrefix = {arXiv},
  url          = {[URL_of_the_paper](https://github.com/ugiat/NERCat/blob/main/Catalan_GLiNER_Paper.pdf)}
}
```

## 🙏 Acknowledgements

- [GLiNER original authors](https://github.com/urchade/GLiNER)
- [Knowledgator original authors](https://github.com/Knowledgator)

## 🤝 Contributing
For any questions, suggestions, or contributions, feel free to open an issue or submit a pull request on this repository. You can also send a e-mail to: ugiat@ugiat.com

## Join Our Community on Discord 🚀
[![Discord](https://img.shields.io/discord/YOUR_SERVER_ID?label=Discord&logo=discord&color=blue)](https://discord.gg/YaghXngv)

