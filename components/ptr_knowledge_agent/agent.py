"""
PTR Knowledge Base Agent implementation.
"""

import os
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .models import PTRKnowledgeConfig

class PTRKnowledgeAgent:
    """Agent for answering questions about PTR using Astra DB knowledge base."""
    
    def __init__(self, config: PTRKnowledgeConfig):
        """Initialize the agent with configuration."""
        self.config = config
        self.llm = self._setup_llm()
        self.vector_store = self._setup_vector_store()
        self.chain = self._setup_chain()
        
    def _setup_llm(self) -> ChatOpenAI:
        """Set up the language model."""
        return ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _setup_vector_store(self) -> AstraDBVectorStore:
        """Set up the vector store connection."""
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small",
            dimensions=1024  # Explicitly set to match your Astra DB collection
        )
        
        # First, try to delete the existing collection if it exists
        try:
            store = AstraDBVectorStore(
                embedding=embeddings,
                collection_name="ptr_knowledgebase",
                api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
                token=os.getenv("ASTRA_DB_TOKEN"),
                namespace="default_keyspace"
            )
            store.collection.delete()
        except Exception:
            pass  # Collection might not exist, which is fine
        
        # Create a new collection with the correct settings
        return AstraDBVectorStore(
            embedding=embeddings,
            collection_name="ptr_knowledgebase",
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_TOKEN"),
            namespace="default_keyspace"
        )
    
    def _setup_chain(self):
        """Set up the RAG chain."""
        # System message template
        system_template = self.config.system_message or """
        You are a PTR Knowledge Base AI Assistant. Your role is to answer questions about PTR Technology, 
        products, services, and documentation using information from the Astra DB knowledge base.
        
        Guidelines:
        1. Only answer questions related to PTR
        2. Use information from the knowledge base
        3. Be clear and professional
        4. Use bullet points for long answers
        5. If information is not found, say so politely
        """
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Create the RAG chain
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def get_question(input_dict):
            return input_dict["question"]
        
        rag_chain = (
            {
                "context": lambda x: retriever.get_relevant_documents(x["question"]) | format_docs,
                "question": get_question,
                "chat_history": lambda x: x["chat_history"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def answer_question(self, question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Answer a question using the knowledge base."""
        if chat_history is None:
            chat_history = []
            
        # Convert chat history to LangChain message format
        messages = []
        for msg in chat_history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Get response from chain
        response = self.chain.invoke({
            "question": question,
            "chat_history": messages
        })
        
        return response 