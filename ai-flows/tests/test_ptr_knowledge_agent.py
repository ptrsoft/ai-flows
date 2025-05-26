import pytest
from unittest.mock import Mock, patch
from langflow.schema import Data
from .knowledge import AstraDBVectorStoreComponent

@pytest.fixture
def mock_astradb_component():
    """Fixture to create a mock AstraDBVectorStoreComponent instance."""
    component = AstraDBVectorStoreComponent()
    component.token = "test_token"
    component.environment = "test"
    component.database_name = "test_db"
    component.collection_name = "test_collection"
    component.api_endpoint = "https://test-db.test-region.apps.astra-test.datastax.com"
    return component

@pytest.fixture
def mock_database():
    """Fixture to create a mock database object."""
    mock_db = Mock()
    mock_db.api_endpoint = "https://test-db.test-region.apps.astra-test.datastax.com"
    mock_db.keyspace = "test_keyspace"
    return mock_db

@pytest.fixture
def mock_collection():
    """Fixture to create a mock collection object."""
    mock_col = Mock()
    mock_col.name = "test_collection"
    return mock_col

def test_get_database_id_static():
    """Test getting database ID from API endpoint."""
    api_endpoint = "https://123e4567-e89b-12d3-a456-426614174000-us-east1.apps.astra-test.datastax.com"
    db_id = AstraDBVectorStoreComponent.get_database_id_static(api_endpoint)
    assert db_id == "123e4567-e89b-12d3-a456-426614174000"

def test_get_database_id_static_invalid():
    """Test getting database ID from invalid API endpoint."""
    api_endpoint = "https://invalid-endpoint.com"
    db_id = AstraDBVectorStoreComponent.get_database_id_static(api_endpoint)
    assert db_id is None

def test_get_keyspace(mock_astradb_component):
    """Test getting keyspace."""
    # Test with keyspace set
    mock_astradb_component.keyspace = "test_keyspace"
    assert mock_astradb_component.get_keyspace() == "test_keyspace"

    # Test with no keyspace
    mock_astradb_component.keyspace = None
    assert mock_astradb_component.get_keyspace() is None

    # Test with whitespace keyspace
    mock_astradb_component.keyspace = "  test_keyspace  "
    assert mock_astradb_component.get_keyspace() == "test_keyspace"

@patch('ai-flows.components.ptr_knowledge_agent.knowledge.DataAPIClient')
def test_get_database_object(mock_client, mock_astradb_component, mock_database):
    """Test getting database object."""
    mock_client.return_value.get_database.return_value = mock_database
    db = mock_astradb_component.get_database_object()
    assert db == mock_database
    mock_client.return_value.get_database.assert_called_once()

@patch('ai-flows.components.ptr_knowledge_agent.knowledge.DataAPIClient')
def test_get_database_object_error(mock_client, mock_astradb_component):
    """Test getting database object with error."""
    mock_client.return_value.get_database.side_effect = Exception("Test error")
    with pytest.raises(ValueError, match="Error fetching database object: Test error"):
        mock_astradb_component.get_database_object()

def test_map_search_type(mock_astradb_component):
    """Test mapping search types."""
    # Test similarity search
    mock_astradb_component.search_type = "Similarity"
    assert mock_astradb_component._map_search_type() == "similarity"

    # Test similarity with score threshold
    mock_astradb_component.search_type = "Similarity with score threshold"
    assert mock_astradb_component._map_search_type() == "similarity_score_threshold"

    # Test MMR search
    mock_astradb_component.search_type = "MMR (Max Marginal Relevance)"
    assert mock_astradb_component._map_search_type() == "mmr"

def test_build_search_args(mock_astradb_component):
    """Test building search arguments."""
    # Test with query
    mock_astradb_component.search_query = "test query"
    mock_astradb_component.search_type = "Similarity"
    mock_astradb_component.number_of_results = 5
    mock_astradb_component.search_score_threshold = 0.5
    args = mock_astradb_component._build_search_args()
    assert args == {
        "query": "test query",
        "search_type": "similarity",
        "k": 5,
        "score_threshold": 0.5
    }

    # Test with filter
    mock_astradb_component.search_query = None
    mock_astradb_component.advanced_search_filter = {"field": "value"}
    args = mock_astradb_component._build_search_args()
    assert args == {
        "n": 4,  # default number_of_results
        "filter": {"field": "value"}
    }

    # Test with no query or filter
    mock_astradb_component.advanced_search_filter = None
    args = mock_astradb_component._build_search_args()
    assert args == {}

@patch('ai-flows.components.ptr_knowledge_agent.knowledge.AstraDBVectorStore')
def test_build_vector_store(mock_vector_store, mock_astradb_component):
    """Test building vector store."""
    mock_astradb_component.embedding_model = Mock()
    mock_astradb_component.embedding_choice = "Embedding Model"
    mock_astradb_component.autodetect_collection = True
    mock_astradb_component.content_field = "test_content"
    mock_astradb_component.ignore_invalid_documents = True
    mock_astradb_component.astradb_vectorstore_kwargs = {"test_param": "test_value"}

    mock_astradb_component.get_database_object = Mock(return_value=mock_database)
    mock_database.list_collection_names.return_value = [mock_astradb_component.collection_name]
    mock_astradb_component.collection_data.return_value = 0

    vector_store = mock_astradb_component.build_vector_store()
    assert vector_store == mock_vector_store.return_value

    mock_vector_store.assert_called_once_with(
        token=mock_astradb_component.token,
        api_endpoint=mock_database.api_endpoint,
        namespace=mock_database.keyspace,
        collection_name=mock_astradb_component.collection_name,
        environment=mock_astradb_component.environment,
        ext_callers=[("langflow", "0.0.0")],
        autodetect_collection=True,
        content_field="test_content",
        ignore_invalid_documents=True,
        embedding=mock_astradb_component.embedding_model,
        test_param="test_value"
    )

def test_get_retriever_kwargs(mock_astradb_component):
    """Test getting retriever kwargs."""
    mock_astradb_component.search_query = "test query"
    mock_astradb_component.search_type = "Similarity"
    mock_astradb_component.number_of_results = 5
    mock_astradb_component.search_score_threshold = 0.5

    kwargs = mock_astradb_component.get_retriever_kwargs()
    assert kwargs == {
        "search_type": "similarity",
        "search_kwargs": {
            "query": "test query",
            "search_type": "similarity",
            "k": 5,
            "score_threshold": 0.5
        }
    }

@patch('ai-flows.components.ptr_knowledge_agent.knowledge.AstraDBVectorStore')
def test_search_documents(mock_vector_store, mock_astradb_component):
    """Test searching documents."""
    mock_astradb_component.search_query = "test query"
    mock_astradb_component.search_type = "Similarity"
    mock_astradb_component.number_of_results = 5

    mock_docs = [Mock()]
    mock_vector_store.return_value.search.return_value = mock_docs

    results = mock_astradb_component.search_documents(mock_vector_store.return_value)
    assert len(results) == 1

    mock_vector_store.return_value.search.assert_called_once_with(
        query="test query",
        search_type="similarity",
        k=5,
        score_threshold=0
    )

def test_search_documents_no_input(mock_astradb_component):
    """Test searching documents with no input."""
    mock_astradb_component.search_query = None
    mock_astradb_component.advanced_search_filter = None

    results = mock_astradb_component.search_documents()
    assert results == [] 
    