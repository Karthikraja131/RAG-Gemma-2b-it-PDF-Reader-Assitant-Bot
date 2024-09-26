from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.schema import Document
from transformers import AutoTokenizer

CACHE_DIR = "huggingface_cache"

class Encoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L12-v2", device="cuda:0"):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=CACHE_DIR,
            model_kwargs={"device": device}
        )

class FaissDb:
    def __init__(self, docs, embedding_function):
        documents = [Document(page_content=doc) for doc in docs]
        self.db = FAISS.from_documents(documents, embedding_function, distance_strategy=DistanceStrategy.COSINE)

    def similarity_search(self, question: str, k: int = 5):
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context

