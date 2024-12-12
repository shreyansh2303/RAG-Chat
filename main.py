import os
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
import chromadb
from chromadb.utils import embedding_functions


os.environ["GROQ_API_KEY"] = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def extract_data(input_file):

    # Load the input file
    loader = PyMuPDFLoader(input_file)
    document = loader.load()

    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(document)

    return [document.page_content.replace('\n', ' ') for document in documents]


documents = extract_data('input.pdf')

chroma_client = chromadb.PersistentClient(path="vectordb")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
collection = chroma_client.get_or_create_collection(name="my_collection", embedding_function=sentence_transformer_ef)

ids = []
for id in range(1,len(documents)+1):
    ids.append(str(id))

collection.add(
    documents=documents,
    ids = ids
)


def get_results(query, num_results):

    results = collection.query(
        query_texts = [query],
        n_results   = num_results
    )
    
    combined_results = ""

    for i, result in enumerate(results['documents'][0]):
        combined_results += f"""Search result {i+1} is as follows: \"{result}\"      """

    return combined_results


llm_client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
)

def get_llm_output(user_query):

    num_results = 3
    results = get_results(user_query, num_results)

    complete_query = f"""
Your task is to answer the user's query solely based on the search results from a document. These search results are from similarity matching the user's query to paragraphs in the document. 
Here are the search results in the order of most similar to least similar: 
 
{results} 
 
Here is the user's query: 
{user_query} 
 
Only give an answer if it can be given from the data in the search results. Summarize what you want to say, don't mention which search result gave you the answer or even the fact that you got your results from the searches.
Try getting the results from the most similar search results (search result 1 is most similar, search result {num_results} is least similar). 
If you can't give a relevant result from the search results, do not make up an answer yourself. Only say exactly this statement word for word: "Sorry, I didn’t understand your question. Do you want to connect with a live agent?" and nothing else. 
"""


    response = llm_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": complete_query,
            }
        ],
        model="llama3-8b-8192",
    )

    return response.choices[0].message.content



if __name__ == "__main__":

    print("\n----------------------\nWelcome to my ChatBot!\n----------------------\n")
    print("Type 'exit' to quit the program.\n")

    while True:
        
        user_query = input("Enter your query: ").strip()
        
        if user_query.lower() == 'exit':
            break
        
        response = get_llm_output(user_query)
        
        print("Response:", response)
        print()

    
    print("Goodbye!")