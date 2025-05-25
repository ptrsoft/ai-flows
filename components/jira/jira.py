import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import httpx
import validators

from langflow.custom import Component
from langflow.io import (
    BoolInput,
    DataInput,
    DropdownInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
    Output,
    StrInput,
    TableInput,
)
from langflow.schema import Data
from langflow.schema.dotdict import dotdict


class JiraAPIComponent(Component):
    display_name = "Jira API Access"
    description = "Make requests to Jira API for issue management. IMPORTANT: Always specify the 'endpoint' parameter (e.g., 'Get Issue', 'Create Issue') for all requests."
    icon = "Jira"
    name = "JiraAPI"

    default_keys = ["endpoint", "issue_key"]

    inputs = [
        DropdownInput(
            name="endpoint",
            display_name="Endpoint",
            options=[
                "Get Issue", 
                "Create Issue", 
                "Add Comment", 
                "Update Issue", 
                "Search Issues",
                "Get Issue Transitions",
                "Transition Issue"
            ],
            info="The Jira API endpoint to use.",
            real_time_refresh=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="issue_key",
            display_name="Issue Key",
            info="The Jira issue key (e.g., PROJECT-123).",
            advanced=False,
            tool_mode=True,
        ),
        MessageTextInput(
            name="project_key",
            display_name="Project Key",
            info="The Jira project key for creating issues.",
            advanced=True,
            tool_mode=True,
        ),
        DropdownInput(
            name="issue_type",
            display_name="Issue Type",
            options=["Bug", "Task", "Story", "Epic", "Sub-task", "Improvement", "New Feature"],
            info="The type of issue to create (optional).",
            advanced=False,
            tool_mode=True,
        ),
        MessageTextInput(
            name="summary",
            display_name="Summary",
            info="The summary/title of the issue.",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="due_date",
            display_name="Due Date",
            info="The due date for the issue in YYYY-MM-DD format.",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="start_date",
            display_name="Start Date",
            info="The start date for the issue in YYYY-MM-DD format.",
            advanced=True,
            tool_mode=True,
        ),
        DropdownInput(
            name="priority",
            display_name="Priority",
            options=["Highest", "High", "Medium", "Low", "Lowest"],
            info="The priority level of the issue.",
            advanced=True,
            tool_mode=True,
        ),
        DropdownInput(
            name="jira_status",
            display_name="Status",
            options=["To Do", "In Progress", "Done", "Blocked", "In Review", "Ready for Review", "Reopened"],
            info="The status of the issue.",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="parent_epic",
            display_name="Parent Epic",
            info="The key of the parent epic (e.g., PROJECT-123).",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="original_estimate",
            display_name="Original Estimate",
            info="The original time estimate (e.g., 4h, 2d, 30m).",
            advanced=True,
            tool_mode=True,
        ),
        MultilineInput(
            name="ic_description",
            display_name="Description",
            info="The description of the issue or comment.",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="jql",
            display_name="JQL Query",
            info="Jira Query Language for searching issues.",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="transition_id",
            display_name="Transition ID",
            info="The ID of the transition to perform on an issue.",
            advanced=True,
            tool_mode=True,
        ),
        TableInput(
            name="fields",
            display_name="Custom Fields",
            info="Additional fields to include in the request.",
            table_schema=[
                {
                    "name": "key",
                    "display_name": "Field Key",
                    "type": "str",
                    "description": "Field key (e.g., customfield_10001)",
                },
                {
                    "name": "value",
                    "display_name": "Value",
                    "description": "Field value",
                },
            ],
            value=[],
            input_types=["Data"],
            advanced=True,
        ),
        StrInput(
            name="jira_base_url",
            display_name="Jira Base URL",
            info="The base URL of your Jira instance (e.g., https://your-domain.atlassian.net).",
            advanced=False,
        ),
        StrInput(
            name="jira_username",
            display_name="Jira Username/Email",
            info="Your Jira username or email address.",
            advanced=False,
        ),
        StrInput(
            name="jira_api_token",
            display_name="Jira API Token",
            info="Your Jira API token.",
            advanced=False,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            value=10,
            info="The timeout to use for the request in seconds.",
            advanced=True,
        ),
        BoolInput(
            name="include_metadata",
            display_name="Include Metadata",
            value=False,
            info="Include additional metadata in the response.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="make_request"),
    ]

    def _process_custom_fields(self, fields: Any) -> dict:
        """Process the custom fields input into a valid dictionary."""
        if fields is None:
            return {}
        
        if isinstance(fields, dict):
            return fields
        
        if isinstance(fields, list):
            processed_fields = {}
            try:
                for item in fields:
                    if not isinstance(item, dict) or "key" not in item or "value" not in item:
                        continue
                    
                    key = item["key"]
                    value = item["value"]
                    
                    # Try to parse JSON values
                    try:
                        if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                            value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                    
                    processed_fields[key] = value
                    
            except (KeyError, TypeError, ValueError) as e:
                self.log(f"Failed to process custom fields: {e}")
                return {}
                
            return processed_fields
        
        return {}

    def _build_auth(self) -> tuple:
        """Build the authentication tuple for Jira API."""
        username = self.jira_username or os.getenv("JIRA_USERNAME", "")
        api_token = self.jira_api_token or os.getenv("JIRA_API_TOKEN", "")
        
        if not username or not api_token:
            raise ValueError("Jira username and API token are required")
            
        return (username, api_token)

    def _build_url(self, endpoint: str, issue_key: Optional[str] = None) -> str:
        """Build the URL for the Jira API request."""
        base_url = self.jira_base_url or os.getenv("JIRA_BASE_URL", "")
        
        if not base_url:
            raise ValueError("Jira base URL is required")
            
        if not base_url.endswith("/"):
            base_url += "/"
            
        # Remove protocol if present to validate
        url_to_validate = base_url
        if "://" in url_to_validate:
            url_to_validate = url_to_validate.split("://")[1]
            
        if not validators.domain(url_to_validate.split("/")[0]):
            raise ValueError(f"Invalid Jira base URL: {base_url}")
            
        api_path = "rest/api/3/"
        
        # Validate endpoint is supported
        supported_endpoints = [
            "Get Issue", "Create Issue", "Add Comment", "Update Issue", 
            "Search Issues", "Get Issue Transitions", "Transition Issue",
            "Get Project Issue Types"
        ]
        
        if endpoint not in supported_endpoints:
            supported_list = ", ".join(f"'{ep}'" for ep in supported_endpoints)
            raise ValueError(f"Unsupported endpoint: '{endpoint}'. Must be one of: {supported_list}")
        
        if endpoint == "Get Issue":
            if not issue_key:
                raise ValueError("Issue key is required for Get Issue endpoint")
            return f"{base_url}{api_path}issue/{issue_key}"
            
        elif endpoint == "Create Issue":
            return f"{base_url}{api_path}issue"
            
        elif endpoint == "Add Comment":
            if not issue_key:
                raise ValueError("Issue key is required for Add Comment endpoint")
            return f"{base_url}{api_path}issue/{issue_key}/comment"
            
        elif endpoint == "Update Issue":
            if not issue_key:
                raise ValueError("Issue key is required for Update Issue endpoint")
            return f"{base_url}{api_path}issue/{issue_key}"
            
        elif endpoint == "Search Issues":
            return f"{base_url}{api_path}search"
            
        elif endpoint == "Get Issue Transitions":
            if not issue_key:
                raise ValueError("Issue key is required for Get Issue Transitions endpoint")
            return f"{base_url}{api_path}issue/{issue_key}/transitions"
            
        elif endpoint == "Transition Issue":
            if not issue_key:
                raise ValueError("Issue key is required for Transition Issue endpoint")
            return f"{base_url}{api_path}issue/{issue_key}/transitions"
            
        elif endpoint == "Get Project Issue Types":
            if not issue_key:  # Using issue_key parameter for project_key
                raise ValueError("Project key is required for Get Project Issue Types endpoint")
            return f"{base_url}{api_path}issuetype/project?projectIdOrKey={issue_key}"
            
        else:
            # This should never happen due to the validation above, but keeping as a fallback
            raise ValueError(f"Unsupported endpoint: {endpoint}")

    def _format_description(self, text: str) -> dict:
        """Format text into Jira's Atlassian Document Format."""
        # Ensure we're using the provided text, not the component description
        if not text or not isinstance(text, str):
            self.log(f"Warning: Invalid description text provided: {text}")
            text = "No description provided"
            
        return {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        }

    async def get_transition_id(self, issue_key: str, status_name: str) -> Optional[str]:
        """Fetch the transition ID for a given status name on an issue."""
        url = self._build_url("Get Issue Transitions", issue_key)
        auth = self._build_auth()
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                auth=auth,
                timeout=self.timeout
            )
            response.raise_for_status()
            transitions = response.json().get("transitions", [])
            for t in transitions:
                if t["name"].lower() == status_name.lower():
                    return t["id"]
        return None

    def _build_request_data(self, endpoint: str) -> dict:
        """Build the request data based on the endpoint."""
        data = {}
        
        if endpoint == "Create Issue":
            project_key = self.project_key.strip() if self.project_key else None
            issue_type = self.issue_type.strip() if self.issue_type else None
            self.log(f"Received project_key: '{project_key}' (type: {type(project_key)})")
            self.log(f"Received issue_type: '{issue_type}' (type: {type(issue_type)})")

            if not project_key:
                raise ValueError("Project key is required for creating an issue")
            if not self.summary:
                raise ValueError("Summary is required for creating an issue")

            data = {
                "fields": {
                    "project": {
                        "key": project_key
                    },
                    "summary": self.summary
                }
            }

            # Add issue type if provided
            if issue_type:
                # Use default issue types if we can't fetch project-specific ones
                valid_type_names = ["bug", "task", "story", "epic", "sub-task", "improvement", "new feature"]
                data["fields"]["issuetype"] = {"name": issue_type}

            if self.ic_description:
                data["fields"]["description"] = self._format_description(self.ic_description)
            if self.due_date:
                data["fields"]["duedate"] = self.due_date
            if self.start_date:
                data["fields"]["customfield_10015"] = self.start_date
            if self.priority:
                # Ensure priority is sent as an object
                data["fields"]["priority"] = {"name": self.priority.strip()} if isinstance(self.priority, str) else self.priority
            if self.parent_epic:
                data["fields"]["parent"] = {"key": self.parent_epic}
            if self.original_estimate:
                data["fields"]["timetracking"] = {
                    "originalEstimate": self.original_estimate
                }
            self.log(f"Final payload for create issue: {json.dumps(data, indent=2)}")
            
        elif endpoint == "Add Comment":
            if not self.ic_description:
                raise ValueError("Description is required for adding a comment")
                
            # Only include the comment body
            data = {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.ic_description
                                }
                            ]
                        }
                    ]
                }
            }
            
        elif endpoint == "Update Issue":
            # Start with an empty update object
            update_data = {}
            
            # Debug logging for summary input
            self.log(f"Raw summary input: {repr(self.summary)}")
            self.log(f"Summary type: {type(self.summary)}")
            
            # Handle summary update separately
            if self.summary is not None:
                if not isinstance(self.summary, str):
                    self.log(f"Invalid summary type: {type(self.summary)}")
                    raise ValueError(f"Summary must be a string value, got {type(self.summary)}")
                summary = self.summary.strip()
                self.log(f"Stripped summary: {repr(summary)}")
                if not summary:
                    self.log("Empty summary after stripping")
                    raise ValueError("Summary cannot be empty")
                update_data["summary"] = summary
            
            # Handle priority update
            if self.priority is not None:
                if not isinstance(self.priority, str):
                    self.log(f"Invalid priority type: {type(self.priority)}")
                    raise ValueError(f"Priority must be a string value, got {type(self.priority)}")
                priority = self.priority.strip()
                self.log(f"Setting priority to: {priority}")
                update_data["priority"] = {"name": priority}
            
            # Build the final request data
            data = {}
            if update_data:
                data["fields"] = update_data
            
            # Log the complete request data
            self.log("Complete request data:")
            self.log(json.dumps(data, indent=2))
            
            # Validate the final data structure
            if not data or not data.get("fields"):
                self.log("No valid fields in request data")
                raise ValueError("No valid fields provided for update")
            
        elif endpoint == "Search Issues":
            if not self.jql:
                raise ValueError("JQL query is required for searching issues")
                
            data = {
                "jql": self.jql,
                "maxResults": 50
            }
            
        elif endpoint == "Transition Issue":
            # Now, transition_id is always set by make_request
            if not self.transition_id:
                raise ValueError("Transition ID is required for transitioning an issue")
            data = {
                "transition": {
                    "id": self.transition_id
                }
            }
            # If a status is specified, add it to the transition data
            if self.jira_status:
                data["fields"] = {
                    "status": {
                        "name": self.jira_status
                    }
                }
                
        # Add custom fields to the request data
        custom_fields = self._process_custom_fields(self.fields)
        if custom_fields and "fields" in data:
            data["fields"].update(custom_fields)
        elif custom_fields:
            if endpoint in ["Get Issue", "Get Issue Transitions"]:
                # These endpoints don't support a request body with fields
                pass
            else:
                data["fields"] = custom_fields
                
        return data

    def _get_request_method(self, endpoint: str) -> str:
        """Get the HTTP method for the endpoint."""
        if endpoint in ["Get Issue", "Search Issues", "Get Issue Transitions"]:
            return "GET"
        elif endpoint in ["Create Issue", "Add Comment"]:
            return "POST"
        elif endpoint == "Update Issue":
            return "PUT"
        elif endpoint == "Transition Issue":
            return "POST"
        else:
            return "GET"

    async def verify_update(self, issue_key: str, expected_summary: str) -> bool:
        """Verify if the issue was updated successfully."""
        try:
            issue_url = self._build_url("Get Issue", issue_key)
            auth = self._build_auth()
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    issue_url,
                    headers=headers,
                    auth=auth,
                    timeout=self.timeout
                )
                response.raise_for_status()
                issue_data = response.json()
                
                current_summary = issue_data.get("fields", {}).get("summary", "")
                self.log(f"Verification - Current summary: '{current_summary}'")
                self.log(f"Verification - Expected summary: '{expected_summary}'")
                
                return current_summary == expected_summary
        except Exception as e:
            self.log(f"Verification failed: {str(e)}")
            return False

    async def make_request(self) -> Data:
        """Make a request to the Jira API."""
        # Debug logging for all attributes
        self.log("Making Jira API request with the following configuration:")
        self.log(f"Endpoint: {self.endpoint}")
        self.log(f"Base URL: {self.jira_base_url}")
        self.log(f"Username: {self.jira_username}")
        self.log(f"API Token: {'*' * len(self.jira_api_token) if self.jira_api_token else 'Not set'}")
        
        endpoint = self.endpoint
        issue_key = self.issue_key
        timeout = self.timeout
        include_metadata = self.include_metadata

        # If endpoint is missing but issue_key is provided, default to Get Issue
        if not endpoint and issue_key:
            endpoint = "Get Issue"
            self.log(f"No endpoint specified but issue_key provided. Defaulting to 'Get Issue' endpoint.")

        # Validate that endpoint is provided
        if not endpoint:
            return Data(
                data={
                    "error": "Endpoint parameter is required. Please specify which Jira API endpoint to use.",
                    "status_code": 400
                }
            )

        # Validate credentials before making the request
        if not self.jira_base_url or not self.jira_username or not self.jira_api_token:
            return Data(
                data={
                    "error": "Missing credentials. Please provide Jira base URL, username/email, and API token.",
                    "status_code": 401
                }
            )

        try:
            url = self._build_url(endpoint, issue_key)
            method = self._get_request_method(endpoint)
            auth = self._build_auth()
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            # Only build request data for POST/PUT
            data = {}
            if method in ["POST", "PUT"]:
                data = self._build_request_data(endpoint)
                
            # For GET requests with search parameters
            params = {}
            if endpoint == "Search Issues":
                params = {"jql": self.jql, "maxResults": 50}
                
            self.log(f"Making {method} request to {url}")
            self.log(f"Request data: {json.dumps(data, indent=2)}")
            
            # Add retry logic for update operations
            max_retries = 3
            retry_delay = 1  # seconds
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient() as client:
                        if method == "GET":
                            response = await client.request(
                                method,
                                url,
                                headers=headers,
                                auth=auth,
                                params=params,
                                timeout=timeout
                            )
                        else:
                            response = await client.request(
                                method,
                                url,
                                headers=headers,
                                auth=auth,
                                json=data,
                                timeout=timeout
                            )
                        
                        # Handle 204 No Content response
                        if response.status_code == 204:
                            self.log("Received 204 No Content response - update successful")
                            return Data(data={
                                "status": "success",
                                "message": "Update successful",
                                "status_code": 204,
                                "source": url
                            })
                        
                        # For update operations, verify the change
                        if endpoint == "Update Issue" and self.summary:
                            self.log("Verifying update...")
                            is_updated = await self.verify_update(issue_key, self.summary.strip())
                            if is_updated:
                                self.log("Update verified successfully")
                                # If update was successful, return success even if we got a 500
                                if response.status_code == 500:
                                    return Data(
                                        data={
                                            "info": "The issue summary was successfully updated, but Jira returned a technical error (500). This is a known Jira issue and you can safely ignore this message.",
                                            "status_code": 200,  # Return 200 since update was successful
                                            "updated_summary": self.summary.strip()
                                        }
                                    )
                            else:
                                self.log("Update verification failed")
                                if attempt < max_retries - 1:
                                    self.log(f"Retrying update (attempt {attempt + 1}/{max_retries})...")
                                    await asyncio.sleep(retry_delay)
                                    continue
                        
                        response.raise_for_status()
                        
                        # Only try to parse JSON if we have content
                        if response.content:
                            result = response.json()
                            metadata = {
                                "source": url,
                                "result": result
                            }
                        else:
                            metadata = {
                                "source": url,
                                "result": None
                            }
                        
                        if include_metadata:
                            metadata.update({
                                "status_code": response.status_code,
                                "headers": dict(response.headers),
                                "request_method": method,
                                "request_data": data if method in ["POST", "PUT"] else None
                            })
                        
                        return Data(data=metadata)
                        
                except httpx.HTTPStatusError as e:
                    last_error = e
                    if e.response.status_code == 500 and endpoint == "Update Issue" and attempt < max_retries - 1:
                        self.log(f"Received 500 error, retrying (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(retry_delay)
                        continue
                    raise
                    
            # If we've exhausted all retries, handle the last error
            if last_error:
                raise last_error
                
        except httpx.HTTPStatusError as e:
            self.log("=== HTTP Error Details ===")
            self.log(f"Error: {str(e)}")
            self.log(f"Status code: {e.response.status_code}")
            self.log(f"Response headers: {dict(e.response.headers)}")
            self.log(f"Response text: {e.response.text}")
            self.log(f"Request data: {json.dumps(data, indent=2)}")
            
            error_data = {
                "error": f"HTTP Error: {str(e)}",
                "status_code": e.response.status_code,
                "raw_response": e.response.text,
                "headers": dict(e.response.headers)
            }
            
            try:
                error_json = e.response.json()
                error_data["details"] = error_json
                self.log(f"Error JSON: {json.dumps(error_json, indent=2)}")
                
                # Check for specific Jira error messages
                if isinstance(error_json, dict):
                    error_messages = error_json.get("errorMessages", [])
                    if error_messages:
                        error_data["error"] = f"Jira API Error: {error_messages[0]}"
                        self.log(f"Jira error message: {error_messages[0]}")
            except Exception as json_err:
                self.log(f"Failed to parse error JSON: {json_err}")
                error_data["details"] = f"Could not decode JSON: {json_err}"
            
            return Data(data=error_data)
        except Exception as e:
            self.log(f"Unexpected error: {str(e)}")
            return Data(
                data={
                    "error": str(e),
                    "status_code": 500
                }
            )

    async def get_project_issue_types(self, project_key: str) -> List[Dict[str, Any]]:
        """Fetch valid issue types for a specific project."""
        try:
            # Validate required attributes
            if not hasattr(self, 'jira_base_url') or not self.jira_base_url:
                self.jira_base_url = os.getenv("JIRA_BASE_URL", "")
            if not hasattr(self, 'jira_username') or not self.jira_username:
                self.jira_username = os.getenv("JIRA_USERNAME", "")
            if not hasattr(self, 'jira_api_token') or not self.jira_api_token:
                self.jira_api_token = os.getenv("JIRA_API_TOKEN", "")
            if not hasattr(self, 'timeout'):
                self.timeout = 10  # Default timeout value

            # Validate credentials
            if not self.jira_base_url or not self.jira_username or not self.jira_api_token:
                self.log("Missing required credentials for Jira API")
                return []

            # Validate project key
            if not project_key:
                self.log("Project key is required")
                return []

            url = self._build_url("Get Project Issue Types", project_key)
            auth = self._build_auth()
            
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=headers,
                    auth=auth,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            self.log(f"HTTP Error fetching project issue types: {str(e)}")
            self.log(f"Response status: {e.response.status_code}")
            self.log(f"Response text: {e.response.text}")
            return []
        except Exception as e:
            self.log(f"Error fetching project issue types: {str(e)}")
            return []
        





















        