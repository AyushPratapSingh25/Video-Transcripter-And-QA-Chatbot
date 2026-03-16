from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#Initialize vector store
vectorstore = PineconeVectorStore(
    index_name="steve-jobs-podcast", embedding=embeddings
)
# Initialize chat model
model = init_chat_model("qwen2.5:3b", model_provider="ollama")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant transcript segments from a YouTube video to answer the user's question."""
    # Retrieve top 4 most similar documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)
    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    # Return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = """
    You are an AI assistant that answers questions about a specific YouTube video.

    The video transcript has already been stored in a vector database.

    You MUST use the tool retrieve_context to get transcript sections before answering.

    Rules:
    1. Always call retrieve_context with the user's question.
    2. Read the retrieved transcript.
    3. Answer ONLY using the retrieved transcript.
    4. Cite the source provided.

    Never ask the user for a video link or video ID.
    """
    
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # Build messages list
    messages = [{"role": "user", "content": query}]
    
    # Invoke the agent
    response = agent.invoke({"messages": messages})
    
    # Extract the answer from the last AI message
    answer = response["messages"][-1].content
    
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="What is the main topic of the video?")
    print(result)
    

