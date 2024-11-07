import gradio as gr
from transformers import pipeline
import pdfplumber
import os
import re
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

openai_api_key = os.getenv("OPENAI_API_KEY")


# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings()

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Directory where your PDFs are stored
PDF_DIRECTORY = "Standards/"  # Folder containing all your PDF files

# Mapping of standards keywords to their respective PDF files
standards_mapping = {
    "ISO 27001": "ISO_IEC_27001.pdf",
    "ISO 27701": "ISO_IEC_27701.pdf",
    "GDPR": "GDPR.pdf",
    "Artificial Intelligence Act": "Artificial_Intelligence_Act.pdf",
    "Chips Act": "Chips_Act.pdf",
    "Chips Act Annex": "Chips_Act_Annex.pdf",
    "CIS Controls v8": "CIS_Controls_v8_Guide.pdf",
    "CIS Controls 7.1": "CIS_Controls_Version_7_1.pdf",
    "CSA Security Guidance": "csa_security_guidance_v4.0.pdf",
    "Cyber Resilience Act": "Cyber_Resilience_Act.pdf",
    "IEC 62443-2-1": "IEC62443-2-1.pdf",
    "ISO 15408-1": "ISO_IEC_15408-1.pdf",
    "ISO 15408-2": "ISO_IEC_15408-2.pdf",
    "ISO 15408-3": "ISO_IEC_15408-3.pdf",
    "NIST CSWP 2018": "NIST.CSWP.04162018.pdf",
    "NIST CSWP 29": "NIST.CSWP.29.pdf",
    "NIST SP 800-53r5": "NIST.SP.800-53r5.pdf"
}

# Function to find the appropriate PDF based on query keywords
def find_pdf_from_query(query):
    for keyword, pdf_file in standards_mapping.items():
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, query, re.IGNORECASE):
            logging.info(f"Found keyword '{keyword}' in query.")
            return os.path.join(PDF_DIRECTORY, pdf_file)
    logging.warning("No relevant standard found in query.")
    return None

# Function to extract text from a PDF in chunks
def extract_pdf_text_in_chunks(pdf_path, chunk_size=1000):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text()
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        return text_splitter.split_text(all_text)

# Function to load documents, create embeddings, and store in FAISS
def create_embeddings_for_documents():
    all_texts = []
    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            text_chunks = extract_pdf_text_in_chunks(pdf_path)
            all_texts.extend(text_chunks)  # Collect all text chunks

    # Generate embeddings for all texts and create FAISS vector store
    vector_store = FAISS.from_texts(all_texts, embedding=embedding_model)
    return vector_store

# Create embeddings when the app starts
vector_store = create_embeddings_for_documents()

# Initialize RAG chain with a retriever
retriever = vector_store.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo")
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Gradio chatbot function
def chatbot(query):
    pdf_path = find_pdf_from_query(query)
    
    if pdf_path:
        try:
            # Retrieve relevant documents and use the RAG chain for answering
            response = rag_chain.run(query)
            return response['answer']
        except Exception as e:
            logging.error(f"Error reading PDF or generating response: {e}")
            return "There was an error processing your request. Please try again."
    else:
        return "Sorry, I couldn't find any related standard in your query. Please specify a recognized standard (e.g., ISO 27001, GDPR)."

# Gradio interface with updated syntax
interface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question with a standard (e.g., ISO 27001, GDPR)..."),
    outputs="text",
    title="Cybersecurity Standards Chatbot",
    description="Ask questions about various cybersecurity standards like ISO, NIST, GDPR, etc. by mentioning the standard in your query."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
