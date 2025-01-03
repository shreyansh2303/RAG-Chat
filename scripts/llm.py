import os
from groq import Groq


class LLM:
    def __init__(self, model):
        
        try:
            self.llm_client = Groq(
                api_key=os.environ["GROQ_API_KEY"],
            )
        except:
            print("GROQ_API_KEY not found in the environment variables.")
            exit()

        self.model = model


    def __call__(self, system_query, user_query):
        
        response = self.llm_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_query,
                },
                {
                    "role": "user",
                    "content": user_query,
                }
            ],
            model = self.model,
        )

        return response.choices[0].message.content
