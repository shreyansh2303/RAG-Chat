from ragchat import RAGChat


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