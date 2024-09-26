# PDF-Reader-Assistant-Bot using RAG and Gemma-2B

This project builds a PDF-Reader-Assistant-Bot powered by Hugging Face's Gemma-2B model, using Retrieval-Augmented Generation (RAG) to retrieve answers from PDF documents.

## Features

- Upload PDFs and get answers based on document content.
- Utilizes Hugging Face models for both embedding and generating responses.
- Support for multi-column PDF text extraction.

## Requirements

- Python 3.8+
- `pip` for installing dependencies.

## Installation

1. Clone this repository:


``https://github.com/Karthikraja131/RAG-Gemma-2b-it-PDF-Reader-Assitant-Bot.git``



2. Install the dependencies:

``pip install -r requirements.txt``

3. Run the application:

``streamlit run app.py``

#Usage
* Upload your PDFs to the web app.

* Ask any questions, and the bot will retrieve relevant content from the PDFs to generate an accurate response.


#Model Information
This bot uses the following models:

Gemma-2B: Hugging Face's google/gemma-2b model for language generation.
MiniLM: sentence-transformers/all-MiniLM-L12-v2 for embedding PDF content.

#Contributions

* Feel free to open issues or submit pull requests to improve the bot.


