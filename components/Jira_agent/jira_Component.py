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
                "Transition Issue",
                "Create Project",
                "Get Projects",
                "Get Account ID"
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
            options=["Bug", "Task", "Story", "Epic"],
            info="The type of issue to create.",
            advanced=True,
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
        MessageTextInput(
            name="project_name",
            display_name="Project Name",
            info="The name of the project to create.",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="project_template_key",
            display_name="Project Template Key",
            info="The key of the project template to use (e.g., com.atlassian.jira-core-project-templates:jira-core-simplified-process-control).",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="project_lead_account_id",
            display_name="Project Lead Account ID",
            info="The account ID of the project lead.",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="project_description",
            display_name="Project Description",
            info="The description of the project.",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="project_type_key",
            display_name="Project Type Key",
            info="The type of project (e.g., 'software', 'business', 'service_desk').",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="assignee_type",
            display_name="Assignee Type",
            info="The assignee type (e.g., 'PROJECT_LEAD', 'UNASSIGNED').",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="avatar_id",
            display_name="Avatar ID",
            info="The ID of the project avatar.",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="permission_scheme",
            display_name="Permission Scheme ID",
            info="The ID of the permission scheme to use for the project.",
            advanced=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="user_email",
            display_name="User Email",
            info="The email address of the user to get account ID for. This is required only when using the 'Get Account ID' endpoint.",
            advanced=False,
            tool_mode=True,
            required=False,
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
            "Create Project", "Get Projects", "Get Account ID"
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
            
        elif endpoint == "Create Project":
            return f"{base_url}{api_path}project"
            
        elif endpoint == "Get Projects":
            return f"{base_url}{api_path}project/search"
            
        elif endpoint == "Get Account ID":
            return f"{base_url}{api_path}user/search"
            
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint}")

    def _format_description(self, text: str) -> dict:
        """Format text into Jira's Atlassian Document Format."""
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

    def _build_request_data(self, endpoint: str) -> dict:
        """Build the request data based on the endpoint."""
        data = {}
        
        if endpoint == "Get Account ID":
            if not self.user_email:
                raise ValueError("User email is required for getting account ID")
            return {}  # No request body needed for GET request
            
        elif endpoint == "Create Project":
            if not self.project_key:
                raise ValueError("Project key is required for creating a project")
            if not self.project_name:
                raise ValueError("Project name is required for creating a project")
            if not self.project_template_key:
                raise ValueError("Project template key is required for creating a project")
            if not self.project_lead_account_id:
                raise ValueError("Project lead account ID is required for creating a project")
            if not self.project_type_key:
                raise ValueError("Project type key is required for creating a project")

            data = {
                "key": self.project_key,
                "name": self.project_name,
                "projectTypeKey": self.project_type_key,
                "projectTemplateKey": self.project_template_key,
                "leadAccountId": self.project_lead_account_id
            }

            if self.project_description:
                data["description"] = self.project_description

            if self.assignee_type:
                data["assigneeType"] = self.assignee_type

            if self.avatar_id:
                try:
                    data["avatarId"] = int(self.avatar_id)
                except ValueError:
                    raise ValueError("Avatar ID must be a valid integer")

            if self.permission_scheme:
                try:
                    data["permissionScheme"] = int(self.permission_scheme)
                except ValueError:
                    raise ValueError("Permission scheme ID must be a valid integer")

            self.log(f"Final create project payload: {json.dumps(data, indent=2)}")
            
        elif endpoint == "Create Issue":
            self.log(f"Received project_key: {self.project_key}")
            self.log(f"Received issue_type: {self.issue_type} (type: {type(self.issue_type)})")
            self.log(f"Received summary: {self.summary}")
            
            project_key = self.project_key.strip() if self.project_key else None
            issue_type = self.issue_type.strip() if self.issue_type else None
            
            self.log(f"Processed project_key: {project_key}")
            self.log(f"Processed issue_type: {issue_type}")
            
            if not project_key:
                raise ValueError("Project key is required for creating an issue")
            if not self.summary:
                raise ValueError("Summary is required for creating an issue")
            if not issue_type:
                self.log("Issue type is missing or empty")
                raise ValueError("Issue type is required for creating an issue")

            data = {
                "fields": {
                    "project": {"key": project_key},
                    "summary": self.summary,
                    "issuetype": {"name": issue_type},
                }
            }

            if self.ic_description:
                data["fields"]["description"] = self._format_description(self.ic_description)

            if self.due_date:
                data["fields"]["duedate"] = self.due_date

            if self.start_date:
                data["fields"]["customfield_10015"] = self.start_date

            if self.priority:
                data["fields"]["priority"] = {"name": self.priority.strip()}

            if self.parent_epic and issue_type.lower() == "sub-task":
                data["fields"]["parent"] = {"key": self.parent_epic}

            if self.original_estimate:
                data["fields"]["timetracking"] = {"originalEstimate": self.original_estimate}

            custom_fields = self._process_custom_fields(self.fields)
            if custom_fields:
                data["fields"].update(custom_fields)

            self.log(f"Final create issue payload: {json.dumps(data, indent=2)}")
            
        elif endpoint == "Add Comment":
            if not self.ic_description:
                raise ValueError("Description is required for adding a comment")
                
            comment_text = self.ic_description
            self.log(f"Adding comment with text: {comment_text}")
            
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
                                    "text": comment_text
                                }
                            ]
                        }
                    ]
                }
            }
            
        elif endpoint == "Update Issue":
            data = {"fields": {}}

            if self.jira_status:
                raise ValueError("You cannot update status using the 'Update Issue' endpoint. To change status, use the 'Transition Issue' endpoint with the 'jira_status' field.")

            if self.summary:
                data["fields"]["summary"] = self.summary
            
            if self.ic_description:
                data["fields"]["description"] = self._format_description(self.ic_description)
            
            if self.due_date:
                data["fields"]["duedate"] = self.due_date
            
            if self.start_date:
                data["fields"]["customfield_10015"] = self.start_date
            
            if self.priority:
                data["fields"]["priority"] = {"name": self.priority}
            
            if self.parent_epic:
                data["fields"]["parent"] = {"key": self.parent_epic}
            
            if self.original_estimate:
                data["fields"]["timetracking"] = {
                    "originalEstimate": self.original_estimate
                }
                
        elif endpoint == "Search Issues":
            if not self.jql:
                raise ValueError("JQL query is required for searching issues")
                
            data = {
                "jql": self.jql,
                "maxResults": 50
            }
            
        elif endpoint == "Transition Issue":
            if not self.issue_key:
                raise ValueError("issue_key is required for Transition Issue")
            if not self.jira_status:
                raise ValueError("jira_status is required for Transition Issue")
            
            data = {
                "transition": {
                    "id": self.transition_id
                }
            }
            
        custom_fields = self._process_custom_fields(self.fields)
        if custom_fields and "fields" in data:
            data["fields"].update(custom_fields)
        elif custom_fields:
            if endpoint in ["Get Issue", "Get Issue Transitions"]:
                pass
            else:
                data["fields"] = custom_fields
                
        return data

    def _get_request_method(self, endpoint: str) -> str:
        """Get the HTTP method for the endpoint."""
        if endpoint in ["Get Issue", "Search Issues", "Get Issue Transitions", "Get Projects", "Get Account ID"]:
            return "GET"
        elif endpoint in ["Create Issue", "Add Comment", "Transition Issue", "Create Project"]:
            return "POST"
        elif endpoint == "Update Issue":
            return "PUT"
        else:
            return "GET"

    async def make_request(self) -> Data:
        """
        Make a request to the Jira API.
        Now supports updating both fields and status in a single user request.
        If both field updates and jira_status are provided, will perform both operations sequentially.
        """
        try:
            self.log(f"All attributes at start: {self.__dict__}")
            self.log(f"Starting make_request with endpoint: {self.endpoint}")
            self.log(f"self.issue_type: {self.issue_type} (type: {type(self.issue_type)})")

            if not self.issue_type:
                self.issue_type = "Task"
                self.log("Hardcoded issue_type to 'Task' for debugging.")

            endpoint = self.endpoint
            issue_key = self.issue_key
            timeout = self.timeout
            include_metadata = self.include_metadata

            if endpoint == "Get Account ID":
                if not hasattr(self, 'user_email') or not self.user_email:
                    return Data(data={
                        "error": "User email is required for the 'Get Account ID' endpoint. Please provide the email address of the user you want to get the account ID for.",
                        "status_code": 400,
                        "help": "Example: user@example.com"
                    })
                
                if not validators.email(self.user_email):
                    return Data(data={
                        "error": "Invalid email format. Please provide a valid email address.",
                        "status_code": 400,
                        "help": "Example: user@example.com"
                    })
                
                url = self._build_url(endpoint)
                auth = self._build_auth()
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
                
                params = {
                    "query": self.user_email,
                    "maxResults": 1
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        url,
                        headers=headers,
                        auth=auth,
                        params=params,
                        timeout=timeout
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if result and len(result) > 0:
                        account_id = result[0].get("accountId")
                        if account_id:
                            return Data(data={
                                "account_id": account_id,
                                "user_info": result[0],
                                "status_code": 200,
                                "message": f"Successfully found account ID for {self.user_email}"
                            })
                        else:
                            return Data(data={
                                "error": "Account ID not found in response",
                                "status_code": 404,
                                "user_info": result[0],
                                "help": "The user was found but no account ID was returned. Please check if the user has the correct permissions."
                            })
                    else:
                        return Data(data={
                            "error": f"No user found with email: {self.user_email}",
                            "status_code": 404,
                            "help": "Please verify the email address is correct and the user exists in your Jira instance."
                        })

            if endpoint == "Create Project":
                self.log("Creating new project...")
                url = self._build_url(endpoint)
                method = self._get_request_method(endpoint)
                auth = self._build_auth()
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
                data = self._build_request_data(endpoint)
                
                self.log(f"Project creation request details:")
                self.log(f"URL: {url}")
                self.log(f"Method: {method}")
                self.log(f"Data: {json.dumps(data, indent=2)}")
                
                async with httpx.AsyncClient() as client:
                    try:
                        response = await client.post(  # Explicitly using POST method
                            url,
                            headers=headers,
                            auth=auth,
                            json=data,
                            timeout=timeout
                        )
                        response.raise_for_status()
                        result = response.json() if response.text else {"message": "Project created successfully"}
                        return Data(data={
                            "message": "Project created successfully",
                            "status_code": response.status_code,
                            "result": result
                        })
                    except httpx.HTTPStatusError as e:
                        error_data = {
                            "error": f"Failed to create project: {str(e)}",
                            "status_code": e.response.status_code,
                            "request_data": data
                        }
                        try:
                            error_data["details"] = e.response.json()
                        except json.JSONDecodeError:
                            error_data["details"] = e.response.text
                        return Data(data=error_data)
                    except Exception as e:
                        return Data(data={
                            "error": f"Unexpected error while creating project: {str(e)}",
                            "status_code": 500,
                            "request_data": data
                        })

            if endpoint == "Get Account ID":
                if not self.user_email:
                    return Data(data={
                        "error": "User email is required for the 'Get Account ID' endpoint. Please provide the email address of the user you want to get the account ID for.",
                        "status_code": 400,
                        "help": "Example: user@example.com"
                    })
                
                if not validators.email(self.user_email):
                    return Data(data={
                        "error": "Invalid email format. Please provide a valid email address.",
                        "status_code": 400,
                        "help": "Example: user@example.com"
                    })
                
                url = self._build_url(endpoint)
                auth = self._build_auth()
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
                
                params = {
                    "query": self.user_email,
                    "maxResults": 1
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        url,
                        headers=headers,
                        auth=auth,
                        params=params,
                        timeout=timeout
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if result and len(result) > 0:
                        account_id = result[0].get("accountId")
                        if account_id:
                            return Data(data={
                                "account_id": account_id,
                                "user_info": result[0],
                                "status_code": 200,
                                "message": f"Successfully found account ID for {self.user_email}"
                            })
                        else:
                            return Data(data={
                                "error": "Account ID not found in response",
                                "status_code": 404,
                                "user_info": result[0],
                                "help": "The user was found but no account ID was returned. Please check if the user has the correct permissions."
                            })
                    else:
                        return Data(data={
                            "error": f"No user found with email: {self.user_email}",
                            "status_code": 404,
                            "help": "Please verify the email address is correct and the user exists in your Jira instance."
                        })

            # --- NEW LOGIC: Allow combined field and status update ---
            # If endpoint is Update Issue and jira_status is provided, perform both field update and status transition
            if endpoint == "Update Issue" and self.jira_status:
                # 1. Update fields (if any field changes are present)
                field_update_result = None
                update_fields = any([
                    self.summary, self.ic_description, self.due_date, self.start_date,
                    self.priority, self.parent_epic, self.original_estimate, self.fields
                ])
                if update_fields:
                    self.log("Performing field update before status transition...")
                    # Temporarily clear jira_status to avoid ValueError in _build_request_data
                    original_jira_status = self.jira_status
                    self.jira_status = None
                    url = self._build_url("Update Issue", issue_key)
                    method = self._get_request_method("Update Issue")
                    auth = self._build_auth()
                    headers = {
                        "Accept": "application/json",
                        "Content-Type": "application/json"
                    }
                    data = self._build_request_data("Update Issue")
                    async with httpx.AsyncClient() as client:
                        resp = await client.request(
                            method,
                            url,
                            headers=headers,
                            auth=auth,
                            json=data,
                            timeout=timeout
                        )
                        try:
                            resp.raise_for_status()
                            field_update_result = resp.json() if resp.text else {"message": "Field update succeeded"}
                        except Exception as e:
                            field_update_result = {"error": str(e), "raw_response": resp.text}
                    # Restore jira_status
                    self.jira_status = original_jira_status
                # 2. Perform status transition
                self.log("Performing status transition after field update...")
                # Reuse the Transition Issue logic
                # Try to find the transition ID
                transitions_url = self._build_url("Get Issue Transitions", self.issue_key)
                auth = self._build_auth()
                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        transitions_url,
                        headers=headers,
                        auth=auth,
                        timeout=timeout
                    )
                    resp.raise_for_status()
                    transitions = resp.json().get("transitions", [])
                    found = False
                    for t in transitions:
                        if t["name"].lower() == self.jira_status.lower():
                            self.transition_id = t["id"]
                            found = True
                            break
                    if not found:
                        available = [{"name": t["name"], "id": t["id"]} for t in transitions]
                        return Data(data={
                            "error": f"No transition found for status '{self.jira_status}' on issue {self.issue_key}",
                            "status_code": 400,
                            "available_transitions": available,
                            "message": f"Available transitions for issue {self.issue_key}: {', '.join([t['name'] for t in transitions]) if transitions else 'None'}"
                        })
                if not self.transition_id:
                    return Data(data={"error": "Transition ID is required for transitioning an issue", "status_code": 500})
                # Now perform the transition
                transition_url = self._build_url("Transition Issue", self.issue_key)
                transition_data = {"transition": {"id": self.transition_id}}
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        transition_url,
                        headers=headers,
                        auth=auth,
                        json=transition_data,
                        timeout=timeout
                    )
                    try:
                        resp.raise_for_status()
                        transition_result = resp.json() if resp.text else {"message": "Transition succeeded"}
                    except Exception as e:
                        transition_result = {"error": str(e), "raw_response": resp.text}
                # Fetch the issue to confirm the status
                issue_url = self._build_url("Get Issue", self.issue_key)
                async with httpx.AsyncClient() as client:
                    issue_resp = await client.get(
                        issue_url,
                        headers=headers,
                        auth=auth,
                        timeout=timeout
                    )
                    issue_resp.raise_for_status()
                    issue_data = issue_resp.json()
                    current_status = issue_data.get("fields", {}).get("status", {}).get("name")
                if current_status == self.jira_status:
                    return Data(data={
                        "message": f"Fields and status updated. Issue {self.issue_key} is now in status '{self.jira_status}'.",
                        "status_code": 200,
                        "field_update_result": field_update_result,
                        "transition_result": transition_result,
                        "issue": issue_data
                    })
                else:
                    return Data(data={
                        "error": f"Transition attempted but status is '{current_status}' instead of expected '{self.jira_status}'.",
                        "status_code": 500,
                        "field_update_result": field_update_result,
                        "transition_result": transition_result,
                        "issue": issue_data
                    })
            # --- END NEW LOGIC ---

            if endpoint == "Create Issue":
                if not self.project_key:
                    return Data(data={"error": "Project key is required", "status_code": 400})
                if not self.summary:
                    return Data(data={"error": "Summary is required", "status_code": 400})
                if not self.issue_type:
                    return Data(data={"error": "Issue type is required", "status_code": 400})

            if endpoint == "Transition Issue":
                if not self.issue_key:
                    return Data(data={"error": "issue_key is required for Transition Issue", "status_code": 400})
                if not self.jira_status:
                    return Data(data={"error": "jira_status is required for Transition Issue", "status_code": 400})

            if not endpoint and issue_key:
                endpoint = "Get Issue"
                self.log(f"No endpoint specified but issue_key provided. Defaulting to 'Get Issue' endpoint.")
            
            if not endpoint:
                return Data(
                    data={
                        "error": "Endpoint parameter is required. Please specify which Jira API endpoint to use.",
                        "status_code": 400
                    }
                )
            
            url = self._build_url(endpoint, issue_key)
            method = self._get_request_method(endpoint)
            auth = self._build_auth()
            
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            data = {}
            if method in ["POST", "PUT"]:
                data = self._build_request_data(endpoint)
            
            params = {}
            if endpoint == "Search Issues":
                params = {"jql": self.jql, "maxResults": 50}
            
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
                
                response.raise_for_status()
                
                try:
                    result = response.json()
                except Exception:
                    result = {"raw_response": response.text}
                
                if endpoint == "Transition Issue" and self.jira_status:
                    try:
                        issue_url = self._build_url("Get Issue", issue_key)
                        self.log(f"Checking current issue status at {issue_url}")
                        async with httpx.AsyncClient() as client:
                            issue_resp = await client.get(
                                issue_url,
                                headers=headers,
                                auth=auth,
                                timeout=timeout
                            )
                            issue_resp.raise_for_status()
                            issue_data = issue_resp.json()
                            current_status = issue_data.get("fields", {}).get("status", {}).get("name")
                            if current_status == self.jira_status:
                                return Data(data={
                                    "message": f"Issue {issue_key} has been successfully moved to {self.jira_status}.",
                                    "status_code": 200,
                                    "result": issue_data
                                })
                            else:
                                return Data(data={
                                    "error": f"Transition attempted but status is '{current_status}' instead of expected '{self.jira_status}'.",
                                    "status_code": 500,
                                    "result": issue_data
                                })
                    except Exception as e:
                        self.log(f"Error confirming status after transition: {e}")

                metadata = {
                    "source": url,
                    "result": result
                }
                
                if include_metadata:
                    metadata.update({
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "request_method": method,
                        "request_data": data if method in ["POST", "PUT"] else None
                    })
                
                return Data(data=metadata)
                
        except httpx.TimeoutException:
            return Data(
                data={
                    "error": "Request timed out",
                    "status_code": 408
                }
            )
        except httpx.HTTPStatusError as e:
            error_data = {
                "error": f"HTTP Error: {str(e)}",
                "status_code": e.response.status_code
            }
            
            try:
                error_data["details"] = e.response.json()
            except json.JSONDecodeError:
                error_data["details"] = e.response.text

            if endpoint == "Transition Issue" and e.response.status_code == 500:
                self.log("500 error during transition - checking if status actually changed")
                try:
                    issue_url = self._build_url("Get Issue", issue_key)
                    self.log(f"Checking current issue status at {issue_url}")
                    auth = self._build_auth()
                    headers = {
                        "Accept": "application/json",
                        "Content-Type": "application/json"
                    }
                    async with httpx.AsyncClient() as client:
                        issue_resp = await client.get(
                            issue_url,
                            headers=headers,
                            auth=auth,
                            timeout=timeout
                        )
                        issue_resp.raise_for_status()
                        issue_data = issue_resp.json()
                        current_status = issue_data.get("fields", {}).get("status", {}).get("name")
                        self.log(f"Current status: {current_status}, Target status: {self.jira_status}")
                        if current_status == self.jira_status:
                            self.log("Status change verified despite 500 error")
                            return Data(
                                data={
                                    "message": f"Successfully transitioned issue to '{current_status}' despite API error.",
                                    "status_code": 200,
                                    "result": issue_data,
                                    "transition_details": {
                                        "to_status": current_status,
                                        "transition_id": self.transition_id if hasattr(self, 'transition_id') else None
                                    }
                                }
                            )
                except Exception as check_err:
                    self.log(f"Error checking issue status: {check_err}")
                
            return Data(data=error_data)
        except Exception as e:
            self.log(f"Unexpected error in make_request: {str(e)}")
            return Data(
                data={
                    "error": str(e),
                    "status_code": 500,
                    "details": {
                        "endpoint": self.endpoint,
                        "issue_type": self.issue_type,
                        "project_key": self.project_key,
                        "summary": self.summary
                    }
                }
            )