from dotenv import load_dotenv

from scripts.vector_db import Collection
from scripts.llm import LLM
from scripts.utils import extract_data


class RAGChat:
    def __init__(
        self,
        vector_db_path:str,
        llm_model_name:str = "llama3-8b-8192",
        sentence_transformer_model_name:str = "all-mpnet-base-v2"
    ) -> None:

        try:
            load_dotenv()

            self.llm = LLM(model=llm_model_name)
            self.collection = Collection(
                path = vector_db_path,
                sentence_transformer_model_name = sentence_transformer_model_name
            )

        except Exception as e:
            print(f"Failed to initialize RAGChat with the following error: \n{e}")
            raise
        else:
            print(f"Successfully initialized RAGChat")



    def update_database(self, file_name:str) -> None:
        chunks = extract_data(file_name)

        try:
            self.collection.update(chunks)
        except Exception as e:
            print(f"Couldn't extract any data from \"{file_name}\" due to the following error: \n{e}")
        else:
            print(f"Successfully loaded data from \"{file_name}\" into the database.")



    def query(self, user_query:str, num_results:int = 3) -> str:

        results = self.collection.query(user_query, num_results)

        combined_results = ""
        for i, result in enumerate(results):
            combined_results += f"""Search result {i+1} is as follows: \"{result}\"      """


        system_prompt = f"""
Your task is to answer the user's query solely based on the search results from a document. These search results are from similarity matching the user's query to paragraphs in the document. 
Here are the search results in the order of most similar to least similar: 

{combined_results} 
 
Only give an answer if it can be given from the data in the search results. Summarize what you want to say, don't mention which search result gave you the answer or even the fact that you got your results from the searches.
Try getting the results from the most similar search results (search result 1 is most similar, search result {num_results} is least similar). 
If you can't give a relevant result from the search results, do not make up an answer yourself. Only say exactly this statement word for word: "I can't answer your question from the provided sources." and nothing else. 
"""


        response = self.llm(system_prompt, user_query)
        return response





def main():

    agent = RAGChat(
        vector_db_path = "./vector_db",
        llm_model_name = "llama3-8b-8192",
        sentence_transformer_model_name = "all-mpnet-base-v2"
    )

    files_to_process = []
    print("Enter the names of files to process one by one. Type 'done' when you are finished:")
    while True:
        file_name = input("Enter file name: ").strip()
        if file_name.lower() == 'done':
            break
        files_to_process.append(file_name)

    print("\nProcessing the files...\n")
    for file_name in files_to_process:
        agent.update_database(file_name)


    print("\n----------------------\nWelcome to my ChatBot!\n----------------------\n")
    print("Type 'exit' to quit the program.\n")
    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() == 'exit':
            break
        
        response = agent.query(user_query, num_results=3)
        print("Response:", response)
        print()
    print("Goodbye!")



if __name__ == "__main__":
    main()