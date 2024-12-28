from dotenv import load_dotenv

from .extract_data import extract_data_pdf
from .vector_db import Collection
from .llm import LLM


def get_results(collection, query, num_results):

    results = collection.query(
        query_texts = [query],
        n_results   = num_results
    )
    
    combined_results = ""

    for i, result in enumerate(results['documents'][0]):
        combined_results += f"""Search result {i+1} is as follows: \"{result}\"      """

    return combined_results


def get_llm_output(llm, collection, user_query):

    num_results = 3
    results = get_results(collection, user_query, num_results)
    
    system_prompt = f"""
Your task is to answer the user's query solely based on the search results from a document. These search results are from similarity matching the user's query to paragraphs in the document. 
Here are the search results in the order of most similar to least similar: 

{results} 
 
Only give an answer if it can be given from the data in the search results. Summarize what you want to say, don't mention which search result gave you the answer or even the fact that you got your results from the searches.
Try getting the results from the most similar search results (search result 1 is most similar, search result {num_results} is least similar). 
If you can't give a relevant result from the search results, do not make up an answer yourself. Only say exactly this statement word for word: "Sorry, I didn’t understand your question. Do you want to connect with a live agent?" and nothing else. 
"""


    response = llm(system_prompt, user_query)
    return response




def main():

    load_dotenv()

    collection = Collection(
        path = "vector_db",
        sentence_transformer_model_name = "all-mpnet-base-v2"
    )


    input_file = "input.pdf"
    chunks = extract_data_pdf(input_file)
    collection.update(chunks)

    llm = LLM(model="llama3-8b-8192")


    print("\n----------------------\nWelcome to my ChatBot!\n----------------------\n")
    print("Type 'exit' to quit the program.\n")

    while True:
        
        user_query = input("Enter your query: ").strip()
        
        if user_query.lower() == 'exit':
            break
        
        response = get_llm_output(llm, collection, user_query)
        
        print("Response:", response)
        print()

    
    print("Goodbye!")



if __name__ == "__main__":
    main()