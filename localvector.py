import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    api_key=os.environ.get("g***********"),
    # other params...
)
# 1️⃣ Create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@tool
def similarity_search(query: str, k: int = 3):
    """
    Perform a similarity search in the FAISS index.

    Args:
        query (str): The query string to search for.
        k (int): The number of top results to return.

    Returns:
        list: A list of documents matching the query.
    """
    save_path = "faiss_index"
    vectorstore = FAISS.load_local(
        save_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    results = vectorstore.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    tools = [similarity_search]
    llm_with_tools = llm.bind_tools(tools)

    query = "What is arbitration in the context"

    resp=llm_with_tools.invoke(query)
    print(resp)