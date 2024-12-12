# **PDF-Based Chatbot Using AI and Vector Search**

## **Overview**
This project is a chatbot application designed to answer user queries based on the content of a PDF document. By leveraging advanced AI tools, it retrieves contextually relevant information from the document and provides precise answers. If no relevant answer can be found, the chatbot gracefully falls back to a predefined response, ensuring a seamless user experience.

The project combines modern technologies such as vector databases, embedding models, and language models to build an intelligent system that can process and understand textual data from PDFs.

---

## **Features**
### 1. **PDF Content Understanding**
- Extracts text content from PDF documents using advanced parsing tools.
- Processes large PDF documents by splitting them into manageable chunks of text for efficient embedding and retrieval.

### 2. **Intelligent Query Matching**
- Performs semantic similarity search using vector embeddings to find the most relevant paragraphs in the PDF that match the user query.
- Retrieves contextually accurate responses based on document content.

### 3. **Fallback Response**
- If no meaningful information can be derived from the PDF, the chatbot responds with:
  > *"Sorry, I didn’t understand your question. Do you want to connect with a live agent?"*

### 4. **User Interaction**
- Simple command-line interface (CLI) for user queries.
- Continuous interaction loop allowing users to ask multiple questions in one session.

### 5. **Scalable Design**
- Modular implementation for easy updates or integration with new tools and features.
- Persistent vector database for efficient retrieval across multiple runs.

---

## **Technologies Used**
### **Core Tools**
- **LangChain**:
  - **PyMuPDFLoader**: Extracts text from PDF files.
  - **RecursiveCharacterTextSplitter**: Splits text into manageable chunks with overlap to preserve context.
- **ChromaDB**:
  - Vector database for storing and querying document embeddings.
  - Used for similarity search on the PDF content.
- **Groq**:
  - A powerful language model used to generate natural language answers based on search results.

### **Embeddings**
- **Sentence Transformers**: Utilizes the `all-mpnet-base-v2` model for generating high-quality vector embeddings of text.

### **Programming Language**
- Python (for both backend logic and user interaction).

---

## **How It Works**
### 1. **PDF Processing**
- The system reads the provided PDF, extracts its content, and processes it into chunks of text. Each chunk is stored in a vector database using embeddings.

### 2. **Query Handling**
- When a user inputs a query, the chatbot:
  1. Searches the vector database to find the top matching text chunks.
  2. Combines the results into a prompt for the language model.
  3. Returns the model’s response to the user.

### 3. **Fallback Logic**
- If no results in the database are relevant enough to form a valid response, the chatbot responds with a predefined fallback message.

---
