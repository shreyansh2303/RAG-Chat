from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader


def extract_data_pdf(input_file, chunk_size=1000, chunk_overlap=200):

    loader = PyMuPDFLoader(input_file)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(document)

    return [document.page_content.replace('\n', ' ') for document in documents]
