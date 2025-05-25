import pytest
from unittest.mock import Mock, patch, AsyncMock
from langflow.schema import Data

from components.integrate_knowledgebase.integrate_knowledge import AstraDBVectorStoreComponent
from langflow.base.data.utils import IMG_FILE_TYPES, TEXT_FILE_TYPES
from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
@pytest.fixture
def mock_vector_store():
    """Fixture to create a mock vector store"""
    return Mock()

@pytest.fixture
def component():
    """Fixture to create an instance of AstraDBVectorStoreComponent"""
    return AstraDBVectorStoreComponent()

def test_component_initialization():
    """Test that the component initializes with correct default values"""
    component = AstraDBVectorStoreComponent()
    assert component.display_name == "Astra DB"
    assert component.description == "Ingest and search documents in Astra DB"
    assert component.name == "AstraDB"

@pytest.mark.asyncio
async def test_create_database():
    """Test database creation functionality"""
    component = AstraDBVectorStoreComponent()
    component.token = "test_token"
    component.environment = "test"
    
    with patch('components.integrate_knowledgebase.integrate_knowledge.DataAPIClient') as mock_client:
        mock_admin = Mock()
        mock_client.return_value.get_admin.return_value = mock_admin
        mock_admin.async_create_database = AsyncMock(return_value={"id": "test_db_id"})
        
        result = await component.create_database_api(
            new_database_name="test_db",
            cloud_provider="Google Cloud Platform",
            region="us-central1",
            token="test_token",
            environment="test"
        )
        
        assert result["id"] == "test_db_id"
        mock_admin.async_create_database.assert_called_once()

def test_search_documents():
    """Test document search functionality"""
    component = AstraDBVectorStoreComponent()
    component.search_query = "test query"
    component.number_of_results = 4
    component.search_type = "Similarity"

    # Mock objects mimicking langchain Document structure
    mock_docs = [
        Mock(page_content="Test document 1", metadata={"source": "test1"}),
        Mock(page_content="Test document 2", metadata={"source": "test2"})
    ]

    with patch.object(component, 'build_vector_store') as mock_build:
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_docs
        mock_build.return_value = mock_vector_store

        results = component.search_documents()

        assert len(results) == 2
        # Asserting against the text content using .get_text()
        assert results[0].get_text() == "Test document 1"
        assert results[1].get_text() == "Test document 2"

def test_add_documents_to_vector_store():
    """Test document ingestion functionality"""
    component = AstraDBVectorStoreComponent()
    component.collection_name = "test_collection"
    component.ingest_data = [
        Data(content="Test document 1", metadata={"source": "test1"}),
        Data(content="Test document 2", metadata={"source": "test2"})
    ]
    
    mock_vector_store = Mock()
    
    with patch.object(component, 'get_database_object') as mock_db:
        mock_database = Mock()
        mock_collection = Mock()
        mock_db.return_value = mock_database
        mock_database.get_collection.return_value = mock_collection
        
        component._add_documents_to_vector_store(mock_vector_store)
        
        mock_vector_store.add_documents.assert_called_once()
        assert len(mock_vector_store.add_documents.call_args[0][0]) == 2

def test_error_handling():
    """Test error handling in component"""
    component = AstraDBVectorStoreComponent()

    # Mock DataAPIClient to raise an exception when get_database is called
    with patch('components.integrate_knowledgebase.integrate_knowledge.DataAPIClient') as mock_client:
        mock_client.return_value.get_database.side_effect = Exception("Simulated database error")
        component.token = "test_token" # Provide a token so get_database_object is called
        component.api_endpoint = "test_endpoint" # Provide an endpoint

        with pytest.raises(ValueError) as exc_info:
            # Now targeting get_database_object which raises ValueError on error
            component.get_database_object()

        # Check if the ValueError contains the expected message
        assert "Error fetching database object" in str(exc_info.value)


    