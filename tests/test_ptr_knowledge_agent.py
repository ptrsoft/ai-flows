"""Tests for the PTR Knowledge Base Agent."""

import os
import pytest
from unittest.mock import patch, MagicMock
from components.ptr_knowledge_agent import PTRKnowledgeAgent, PTRKnowledgeConfig

# Mock environment variables
@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "",
        "ASTRA_DB_API_ENDPOINT": "",
        "ASTRA_DB_TOKEN": ""
    }):
        yield

@pytest.fixture
def mock_config():
    """Create a test configuration."""
    return PTRKnowledgeConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1000
    )

@pytest.fixture
def mock_agent(mock_config):
    """Create a test agent with mocked dependencies."""
    with patch("components.ptr_knowledge_agent.agent.ChatOpenAI") as mock_llm, \
         patch("components.ptr_knowledge_agent.agent.OpenAIEmbeddings") as mock_embeddings, \
         patch("components.ptr_knowledge_agent.agent.AstraDBVectorStore") as mock_vector_store, \
         patch("components.ptr_knowledge_agent.agent.RunnablePassthrough") as mock_passthrough, \
         patch("components.ptr_knowledge_agent.agent.StrOutputParser") as mock_parser:
        
        # Set up mock return values
        mock_llm.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        mock_vector_store.return_value = MagicMock()
        mock_passthrough.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        
        # Create the agent
        agent = PTRKnowledgeAgent(mock_config)
        
        # Store mocks for assertions
        agent._mock_llm = mock_llm
        agent._mock_embeddings = mock_embeddings
        agent._mock_vector_store = mock_vector_store
        agent._mock_passthrough = mock_passthrough
        agent._mock_parser = mock_parser
        
        yield agent

def test_agent_initialization(mock_agent, mock_config):
    """Test that the agent initializes correctly."""
    assert mock_agent.config == mock_config
    assert mock_agent.llm is not None
    assert mock_agent.vector_store is not None
    assert mock_agent.chain is not None

def test_llm_setup(mock_agent, mock_config):
    """Test that the LLM is set up correctly."""
    # Verify LLM was initialized with correct parameters
    mock_agent._mock_llm.assert_called_once()
    call_args = mock_agent._mock_llm.call_args[1]
    assert call_args["model_name"] == mock_config.model_name
    assert call_args["temperature"] == mock_config.temperature
    assert call_args["max_tokens"] == mock_config.max_tokens
    assert call_args["openai_api_key"] == os.getenv("OPENAI_API_KEY")

def test_vector_store_setup(mock_agent):
    """Test that the vector store is set up correctly."""
    # Verify embeddings were initialized
    mock_agent._mock_embeddings.assert_called_once()
    call_args = mock_agent._mock_embeddings.call_args[1]
    assert call_args["openai_api_key"] == os.getenv("OPENAI_API_KEY")
    
    # Verify vector store was initialized
    mock_agent._mock_vector_store.assert_called_once()
    call_args = mock_agent._mock_vector_store.call_args[1]
    assert call_args["api_endpoint"] == os.getenv("ASTRA_DB_API_ENDPOINT")
    assert call_args["token"] == os.getenv("ASTRA_DB_TOKEN")
    assert call_args["collection_name"] == "ptr_knowledgebase"
    assert call_args["namespace"] == "default_keyspace"

def test_answer_question(mock_agent):
    """Test that the agent can answer questions."""
    # Mock the chain's invoke method
    mock_agent.chain = MagicMock()
    mock_agent.chain.invoke.return_value = "Test response"
    
    # Test with no chat history
    response = mock_agent.answer_question("What is PTR?")
    assert response == "Test response"
    
    # Test with chat history
    chat_history = [
        {"role": "human", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]
    response = mock_agent.answer_question("What is PTR?", chat_history)
    assert response == "Test response"
    
    # Verify chain was called with correct arguments
    mock_agent.chain.invoke.assert_called()
    call_args = mock_agent.chain.invoke.call_args[0][0]
    assert "question" in call_args
    assert "chat_history" in call_args

def test_agent_config_validation():
    """Test configuration validation."""
    # Test with invalid temperature
    with pytest.raises(ValueError):
        PTRKnowledgeConfig(
            model_name="gpt-3.5-turbo",
            temperature=2.0,  # Invalid temperature
            max_tokens=1000
        )
    
    # Test with invalid max_tokens
    with pytest.raises(ValueError):
        PTRKnowledgeConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=-1  # Invalid max_tokens
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 