import pytest
from unittest.mock import Mock, patch, AsyncMock
from components.Jira_agent.jira_Component import JiraAPIComponent
from langflow.schema import Data

@pytest.fixture
def mock_jira_component():
    return JiraAPIComponent(
        jira_base_url="https://test.atlassian.net",
        jira_username="test@example.com",
        jira_api_token="test-token"
    )

@pytest.mark.asyncio
async def test_get_account_id(mock_jira_component):
    mock_jira_component.user_email = "test@example.com"
    with patch('httpx.AsyncClient.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = [{"accountId": "123", "emailAddress": "test@example.com"}]
        mock_get.return_value = mock_response

        result = await mock_jira_component.make_request()
        
        assert result.data["account_id"] == "123"
        assert result.data["status_code"] == 200

@pytest.mark.asyncio
async def test_create_issue_with_custom_fields(mock_jira_component):
    mock_jira_component.endpoint = "Create Issue"
    mock_jira_component.project_key = "TEST"
    mock_jira_component.summary = "Test Issue"
    mock_jira_component.issue_type = "Bug"
    mock_jira_component.fields = [
        {"key": "customfield_10001", "value": "Custom Value"},
        {"key": "customfield_10002", "value": {"type": "doc", "content": []}}
    ]

    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"id": "123", "key": "TEST-1"}
        mock_post.return_value = mock_response

        result = await mock_jira_component.make_request()
        
        assert result.data["id"] == "123"
        assert result.data["key"] == "TEST-1"

@pytest.mark.asyncio
async def test_transition_issue(mock_jira_component):
    mock_jira_component.endpoint = "Transition Issue"
    mock_jira_component.issue_key = "TEST-1"
    mock_jira_component.jira_status = "In Progress"
    mock_jira_component.transition_id = "31"

    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"id": "123", "key": "TEST-1"}
        mock_post.return_value = mock_response

        result = await mock_jira_component.make_request()
        
        assert result.data["id"] == "123"
        assert result.data["key"] == "TEST-1"

@pytest.mark.asyncio
async def test_combined_update_and_transition(mock_jira_component):
    mock_jira_component.endpoint = "Update Issue"
    mock_jira_component.issue_key = "TEST-1"
    mock_jira_component.summary = "Updated Summary"
    mock_jira_component.jira_status = "In Progress"

    # Mock the transitions request
    with patch('httpx.AsyncClient.get') as mock_get:
        mock_transitions_response = AsyncMock()
        mock_transitions_response.json.return_value = {
            "transitions": [{"id": "31", "name": "In Progress"}]
        }
        mock_get.return_value = mock_transitions_response

        # Mock the update request
        with patch('httpx.AsyncClient.put') as mock_put:
            mock_update_response = AsyncMock()
            mock_update_response.json.return_value = {"id": "123", "key": "TEST-1"}
            mock_put.return_value = mock_update_response

            # Mock the transition request
            with patch('httpx.AsyncClient.post') as mock_post:
                mock_transition_response = AsyncMock()
                mock_transition_response.json.return_value = {"id": "123", "key": "TEST-1"}
                mock_post.return_value = mock_transition_response

                result = await mock_jira_component.make_request()
                
                assert result.data["id"] == "123"
                assert result.data["key"] == "TEST-1"

@pytest.mark.asyncio
async def test_error_handling_invalid_email(mock_jira_component):
    mock_jira_component.endpoint = "Get Account ID"
    mock_jira_component.user_email = "invalid-email"

    result = await mock_jira_component.make_request()
    
    assert result.data["error"] == "Invalid email format. Please provide a valid email address."
    assert result.data["status_code"] == 400

@pytest.mark.asyncio
async def test_error_handling_missing_required_fields(mock_jira_component):
    mock_jira_component.endpoint = "Create Issue"
    # Missing required fields: project_key, summary, issue_type

    result = await mock_jira_component.make_request()
    
    assert "error" in result.data
    assert "Project key is required" in result.data["error"]

@pytest.mark.asyncio
async def test_search_issues_with_jql(mock_jira_component):
    mock_jira_component.endpoint = "Search Issues"
    mock_jira_component.jql = "project = TEST AND priority = High"

    with patch('httpx.AsyncClient.get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "issues": [
                {"id": "123", "key": "TEST-1"},
                {"id": "124", "key": "TEST-2"}
            ]
        }
        mock_get.return_value = mock_response

        result = await mock_jira_component.make_request()
        
        assert len(result.data["issues"]) == 2
        assert result.data["issues"][0]["key"] == "TEST-1"
        assert result.data["issues"][1]["key"] == "TEST-2"

@pytest.mark.asyncio
async def test_create_project(mock_jira_component):
    mock_jira_component.endpoint = "Create Project"
    mock_jira_component.project_key = "TEST"
    mock_jira_component.project_name = "Test Project"
    mock_jira_component.project_type_key = "software"
    mock_jira_component.project_template_key = "com.atlassian.jira-core-project-templates:jira-core-simplified-process-control"
    mock_jira_component.project_lead_account_id = "123"

    with patch('httpx.AsyncClient.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "id": "123",
            "key": "TEST",
            "name": "Test Project"
        }
        mock_post.return_value = mock_response

        result = await mock_jira_component.make_request()
        
        assert result.data["id"] == "123"
        assert result.data["key"] == "TEST"
        assert result.data["name"] == "Test Project" 