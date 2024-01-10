# CV Pipeline

The steps to process a CV are:

1. OCR the CV from pdf to text
1. use LLM to create a structured JSON representation of the CV
1. use the NER model to extract entities from the CV, including:

- ESCO
- PRODUCTS
- frequent VERBS
- frequent NOUNS
