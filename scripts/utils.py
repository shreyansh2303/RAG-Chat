from urllib.parse import urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import requests
from bs4 import BeautifulSoup



    
def is_a_url(file_name: str) -> bool:
    parsed = urlparse(file_name)
    if parsed.scheme and parsed.netloc:
        return True
    else:
        return False


def extract_data_pdf(input_file, chunk_size=1000, chunk_overlap=200):

    try:
        loader = PyMuPDFLoader(input_file)
        document = loader.load()
    except:
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(document)

    return [document.page_content.replace('\n', ' ') for document in documents]


def extract_data_html(url, chunk_size=1000, chunk_overlap=200):

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0'}
    response = requests.get(url, headers=headers)
    text = ""

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        paragraphs = soup.find_all("p")
        for paragraph in paragraphs:
            text += paragraph.get_text(strip=True) + " "

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        chunks = [chunk.replace('\n', ' ') for chunk in chunks]

        return chunks
            
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")
        return []
    

def extract_data(file_name: str, chunk_size=1000, chunk_overlap=200):

    if is_a_url(file_name):
        return extract_data_html(file_name, chunk_size, chunk_overlap)
    else:
        return extract_data_pdf(file_name, chunk_size, chunk_overlap)