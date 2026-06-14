import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from src.data_loader import load_all_documents

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        config_path = os.path.join(persist_dir, "config.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path) and os.path.exists(config_path)):
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            try:
                self.vectorstore.load()
            except (FileNotFoundError, ValueError):
                docs = load_all_documents("data")
                self.vectorstore.build_from_documents(docs)

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

        self.llm = ChatGroq(groq_api_key=groq_api_key, model=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = prompt = f"""
                            You are a strict RAG assistant.
                            Answer the question only using the context below.
                            If the context does not contain the answer, say:
                            "I don't know based on the provided documents."

                            Question:
                            {query}

                            Context:
                            {context}

                            Answer:
                            """
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
