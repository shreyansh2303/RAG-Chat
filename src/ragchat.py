from dotenv import load_dotenv

from vector_db import Collection
from llm import LLM
from utils import extract_data


class RAGChat:
    def __init__(
        self,
        vector_db_path: str,
        llm_model_name: str = "llama3-8b-8192",
        sentence_transformer_model_name: str = "all-mpnet-base-v2"
    ) -> None:

        try:
            load_dotenv()

            self.llm = LLM(model=llm_model_name)
            self.collection = Collection(
                path = vector_db_path,
                sentence_transformer_model_name = sentence_transformer_model_name
            )

        except Exception as e:
            print(f"Failed to initialize RAG-Chat with the following error: \n{e}")
            raise
        else:
            print(f"Successfully initialized RAG-Chat")



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
