import chromadb
from chromadb.utils import embedding_functions


class Collection():
    def __init__(self, path, sentence_transformer_model_name):

        chroma_client = chromadb.PersistentClient(path=path)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=sentence_transformer_model_name
        )

        self.collection = chroma_client.get_or_create_collection(
            name="my_collection",
            embedding_function=sentence_transformer_ef
        )

        self.max_ID = 0

    def update(self, chunks):

        ids = []
        for _ in range(len(chunks)):
            self.max_ID += 1
            ids.append(str(self.max_ID))

        self.collection.add(
            documents=chunks,
            ids = ids
        )

    def query(self, query, num_results):

        results = self.collection.query(
            query_texts = [query],
            n_results   = num_results
        )

        return results['documents'][0]
        