import time
import fitz  # PyMuPDF for PDF handling
import streamlit as st
from model import ChatModel
import util

# Constants
MAX_NEW_TOKENS = 1200  # Default max tokens
K = 5  # Default number of context documents to retrieve

# Model ID from Hugging Face for Gemma 2b
model_id = "google/gemma-2b"

# Load the model and encoder (Hugging Face models, loaded online)
model = ChatModel(model_id=model_id, device="cuda:0")
encoder = util.Encoder(model_name="sentence-transformers/all-MiniLM-L12-v2", device="cuda:0")

# Streamlit UI setup
st.title("PDF-Reader-Assistant-RAG-Bot")
st.write("Ask me anything! I will retrieve answers from the PDF documents.")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

def is_two_column_page(page, threshold=0.5):
    """Check if a PDF page has two columns based on text density distribution."""
    page_width = page.rect.width
    page_height = page.rect.height
    left_column = fitz.Rect(0, 0, page_width / 2, page_height)
    right_column = fitz.Rect(page_width / 2, 0, page_height, page_height)

    # Get text density (character count) for each column
    left_text = page.get_text("text", clip=left_column)
    right_text = page.get_text("text", clip=right_column)

    # Consider it a two-column page if both columns have significant text
    return len(left_text.strip()) > threshold * len(right_text.strip())

def extract_text_from_columns(page, num_columns=2):
    """Extract text from the specified number of columns on a PDF page."""
    text = ""
    page_width = page.rect.width
    column_width = page_width / num_columns

    for col in range(num_columns):
        rect = fitz.Rect(col * column_width, 0, (col + 1) * column_width, page.rect.height)
        column_text = page.get_text("text", clip=rect)
        text += column_text + "\n"
    
    return text

def load_and_process_pdfs(uploaded_files):
    """Load and process PDFs, extracting text meaningfully based on page layout."""
    documents = []
    for uploaded_file in uploaded_files:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            if is_two_column_page(page):
                page_text = extract_text_from_columns(page, num_columns=2)
            else:
                page_text = page.get_text("text")
            documents.append(page_text)
    return documents

# Chat function to handle input and response
def chat(question, DB):
    start_time = time.time()  # Start timing the response generation
    context = DB.similarity_search(question, k=K)
    answer = model.inference(question, context=context, max_new_tokens=MAX_NEW_TOKENS)
    end_time = time.time()  # End timing the response generation
    response_time = end_time - start_time  # Calculate the time taken to generate the response
    return answer, response_time

# Handle PDF processing and embedding
if uploaded_files:
    st.write("Processing uploaded PDF files...")
    start_time = time.time()

    docs = load_and_process_pdfs(uploaded_files)
    DB = util.FaissDb(docs=docs, embedding_function=encoder.embedding_function)

    end_time = time.time()
    embedding_time = end_time - start_time
    st.success(f"Vector embedding completed in {embedding_time:.2f} seconds for {len(uploaded_files)} files.")

    user_input = st.text_input("Enter your question:")
    if user_input:
        response, response_time = chat(user_input, DB)
        st.write("### Assistant Response:")
        st.write(response)
        st.write(f"Response generated in {response_time:.2f} seconds.")
else:
    st.warning("Please upload PDF files to start.")

