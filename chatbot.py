import streamlit as st
import pdfplumber
import camelot
import pytesseract
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Import the Document class for LangChain
from pdf2image import convert_from_path
import os

# Setup for Tesseract OCR (Ensure Tesseract is installed locally)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Ensure the temp directory exists for saving PDFs
if not os.path.exists("temp"):
    os.makedirs("temp")

# ---------------------------
# PDF Content Extraction Functions
# ---------------------------

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    st.write("Extracting text from PDF...")
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf_reader:
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()
    return extracted_text

# Function to extract tables using Camelot
def extract_tables_from_pdf(pdf_file):
    st.write("Extracting tables from PDF...")
    temp_file_path = os.path.join("temp", pdf_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(pdf_file.getbuffer())  # Save uploaded file to disk
    tables = camelot.read_pdf(temp_file_path, pages="all", flavor="stream")
    extracted_tables = []
    for index, table in enumerate(tables):
        table_data = table.df.values.tolist()  # Convert table to list of lists
        extracted_tables.append({"table_index": index + 1, "table_data": table_data})
    return extracted_tables

# Function to extract images and perform OCR
def extract_images_and_ocr(pdf_file):
    st.write("Extracting images and performing OCR...")
    temp_file_path = os.path.join("temp", pdf_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(pdf_file.getbuffer())  # Save uploaded file to disk
    images = convert_from_path(temp_file_path)
    extracted_images = []
    for index, image in enumerate(images):
        ocr_text = pytesseract.image_to_string(image)
        extracted_images.append({"image_index": index + 1, "ocr_text": ocr_text})
    return extracted_images

# ---------------------------
# Processing PDF and Storing Data in Memory
# ---------------------------
def process_pdf_in_memory(uploaded_files):
    all_pdf_data = []
    for uploaded_file in uploaded_files:
        pdf_data = {
            "pdf_name": uploaded_file.name,
            "extracted_text": extract_text_from_pdf(uploaded_file),
            "tables": extract_tables_from_pdf(uploaded_file),
            "images": extract_images_and_ocr(uploaded_file),
        }
        all_pdf_data.append(pdf_data)
    return all_pdf_data

# ---------------------------
# Main Streamlit Application Logic
# ---------------------------

# Initialize extracted data in memory
pdf_data_in_memory = []

# Streamlit PDF uploader for selecting PDF files
st.title("📄 PDF Chatbot with Llama 2 🦙")
st.markdown(
    """
    Welcome to the **PDF Chatbot** powered by **Llama 2**. Upload your PDF and ask questions to extract key insights, tables, and images using OCR.
    """
)

# Adding a sidebar for upload and instructions
with st.sidebar:
    st.header("Instructions:")
    st.write("1. Upload one or more PDF files.")
    st.write("2. Ask any questions related to the PDF.")
    st.write("3. Get answers extracted from text, tables, and images in the PDF.")

# File uploader for the PDF
uploaded_files = st.file_uploader("📂 Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.info("Processing PDF(s)...")
    pdf_data_in_memory = process_pdf_in_memory(uploaded_files)
    st.success("✅ PDFs processed and data is ready for interaction.")

# Ensure a PDF has been processed before moving forward
if pdf_data_in_memory:
    selected_pdf = st.selectbox("Select a PDF to interact with", [pdf['pdf_name'] for pdf in pdf_data_in_memory])

    # Combine all extracted text, tables, and OCR data into one string
    selected_pdf_data = next((pdf for pdf in pdf_data_in_memory if pdf['pdf_name'] == selected_pdf), None)
    if selected_pdf_data:
        st.write("Combining extracted text, tables, and OCR data for embedding...")
        combined_text = selected_pdf_data["extracted_text"]
        
        # Add table data to combined text
        for table in selected_pdf_data["tables"]:
            table_text = '\n'.join(['\t'.join(row) for row in table['table_data']])
            combined_text += "\n" + table_text
        
        # Add OCR text to combined text
        for image in selected_pdf_data["images"]:
            combined_text += "\n" + image['ocr_text']

        # Initialize HuggingFace Embeddings
        st.write("Initializing HuggingFace Embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Use LangChain's text splitter for better chunking of the text
        st.write("Splitting combined text into manageable chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(combined_text)

        # Wrap each chunk in a Document object for LangChain compatibility
        docs = [Document(page_content=chunk) for chunk in chunks]

        # Create the FAISS vector store using LangChain's `from_documents` method
        st.write("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(docs, embeddings)

        st.success("🎉 Vector store created. You can now ask questions!")

        # ---------------------------
        # Question Answering Interaction
        # ---------------------------

        # Input question from the user
        question = st.text_input("Ask a question about the selected PDF")

        if question:
            # Search for relevant chunks in the vector store based on the question
            st.write(f"Searching for relevant content based on question: '{question}'")
            docs = vector_store.similarity_search(question)
            
            # Prepare the input for the invoke method
            input_data = {"input_documents": docs, "question": question}
            
            # Use Llama 2 (via Ollama) to answer the question using the retrieved content
            llm = Ollama(model="llama2")
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Use invoke method to get the result
            result = chain.invoke(input_data)
            
            # Extract the answer from the response
            answer = result.get("output_text", "Sorry, no relevant information found.")

            # Display the question and answer as simple text
            st.write(f"User: {question}")
            st.write(f"Bot: {answer}")



