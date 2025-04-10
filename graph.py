from typing import TypedDict, Annotated, List, Dict, Any
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker


# Define state schema
class AgentState(TypedDict):
    messages: List[Any]
    query: str
    db_results: List[Dict]
    current_db: str
    next_steps: str

# Set up your databases
embeddings = OpenAIEmbeddings()

# Create or load the first database
db1_docs = [
    Document(page_content="Information about topic A", metadata={"source": "db1", "topic": "A"}),
    Document(page_content="More information about topic B", metadata={"source": "db1", "topic": "B"}),
]
db1 = Chroma.from_documents(documents=db1_docs, embedding=embeddings, collection_name="db1")

# Create or load the second database
db2_docs = [
    Document(page_content="Additional facts about topic C", metadata={"source": "db2", "topic": "C"}),
    Document(page_content="Detailed explanation about topic D", metadata={"source": "db2", "topic": "D"}),
]
db2 = Chroma.from_documents(documents=db2_docs, embedding=embeddings, collection_name="db2")

# Create the LLM
llm = ChatOpenAI(model="gpt-4")

# Define agent components
def query_router(state: AgentState) -> str:
    """Determine which database to query or if we should respond"""
    messages = state["messages"]
    response = llm.invoke(
        messages + [
            HumanMessage(
                content=(
                    "Based on the conversation above and the query, determine if you should: "
                    "1. Search database 1 (respond with 'db1') "
                    "2. Search database 2 (respond with 'db2') "
                    "3. Provide a final answer (respond with 'answer') "
                    "Only respond with one of these options: 'db1', 'db2', or 'answer'"
                )
            )
        ]
    )
    return response.content

def search_db1(state: AgentState) -> AgentState:
    """Search the first database"""
    query = state["query"] 
    results = db1.similarity_search(query, k=3)
    docs_info = [{"content": d.page_content, "metadata": d.metadata} for d in results]
    
    return {
        **state,
        "db_results": docs_info,
        "current_db": "db1"
    }

def search_db2(state: AgentState) -> AgentState:
    """Search the second database"""
    query = state["query"]
    results = db2.similarity_search(query, k=3)
    docs_info = [{"content": d.page_content, "metadata": d.metadata} for d in results]
    
    return {
        **state,
        "db_results": docs_info,
        "current_db": "db2"
    }

def generate_response(state: AgentState) -> AgentState:
    """Generate a response based on the database results and query"""
    messages = state["messages"]
    query = state["query"]
    db_results = state["db_results"]
    
    # Format database results for the prompt
    formatted_results = "\n\n".join([
        f"Document {i+1} (Source: {doc['metadata'].get('source')}): {doc['content']}" 
        for i, doc in enumerate(db_results)
    ])
    
    response = llm.invoke(
        messages + [
            HumanMessage(
                content=(
                    f"Based on the following information from the databases and the user query, "
                    f"provide a helpful answer.\n\n"
                    f"User query: {query}\n\n"
                    f"Retrieved information:\n{formatted_results}"
                )
            )
        ]
    )
    
    return {
        **state,
        "messages": messages + [
            HumanMessage(content=query),
            AIMessage(content=response.content)
        ],
        "next_steps": "DONE"
    }

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", query_router)
workflow.add_node("search_db1", search_db1)
workflow.add_node("search_db2", search_db2)
workflow.add_node("generate_response", generate_response)

# Add edges
workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    {
        "db1": "search_db1",
        "db2": "search_db2",
        "answer": "generate_response"
    }
)
workflow.add_edge("search_db1", "generate_response")
workflow.add_edge("search_db2", "generate_response")
workflow.add_edge("generate_response", END)

# Compile the graph
app = workflow.compile()

# Example usage
if __name__ == "__main__":
    # Initialize state
    init_state = {
        "messages": [],
        "query": "Tell me about topic B",
        "db_results": [],
        "current_db": "",
        "next_steps": ""
    }
    
    # Run the graph
    result = app.invoke(init_state)
    
    # Print final messages
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")