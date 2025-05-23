# AI Flows - Multi-Component AI Integration Platform

A comprehensive platform that integrates various AI-powered components for knowledge management and customer support operations.

## Components

### 1. PTR Knowledge Base Assistant
A Python-based knowledge base assistant that integrates with OpenAI and AstraDB to provide intelligent responses to PTR-related queries.

#### Features
- Chat-based interface for knowledge base queries
- OpenAI-powered information extraction and response generation
- AstraDB integration for knowledge storage and retrieval
- Content summarization capabilities
- PTR-specific query filtering
- Comprehensive test suite

#### Usage
```python
from components.ptr_kb_assistant import PTRKBAssistant
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the assistant
assistant = PTRKBAssistant(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    astra_db_config={
        "secure_connect_bundle": os.getenv('ASTRA_DB_SECURE_BUNDLE_PATH'),
        "client_id": os.getenv('ASTRA_DB_CLIENT_ID'),
        "client_secret": os.getenv('ASTRA_DB_CLIENT_SECRET'),
        "keyspace": os.getenv('ASTRA_DB_KEYSPACE')
    }
)

# Process a query
result = assistant.process_query("What are the key features of PTR's 5G solutions?")
print(result)
```

### 2. Outreach API Mock Server
A mock implementation of the Outreach API for testing and development purposes.

#### Features
- Full CRUD operations for leads, accounts, opportunities, sequences, and templates
- In-memory data storage with automatic ID generation
- Relationship management between resources
- Input validation and error handling
- Support for filtering and pagination

#### Running the Mock Server
```bash
uvicorn mockers.outreach.mock_outreach_api:app --reload --port 8000
```

### 3. Email Formation Component
Creates personalized emails using knowledge from various sources including Google Drive, file uploads, and LlamaCloud knowledge base.

#### Features
- LlamaCloud Knowledge Base integration
- Google Drive integration
- File upload support
- OpenAI-powered email generation
- Robust file handling

#### Required Fields for LlamaCloud
- LlamaCloud Index Name
- LlamaCloud Project Name
- LlamaCloud Organization ID
- LlamaCloud API Key

### 4. Kayako MCP Server
A FastAPI server that provides access to Kayako support tickets using the Claude MCP protocol.

#### Features
- Full MCP protocol implementation
- Streaming response support
- Comprehensive ticket data retrieval
- Search functionality
- MAXIS case handling
- Web-based testing interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-flows
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# AstraDB Configuration
ASTRA_DB_SECURE_BUNDLE_PATH=path/to/secure-connect-bundle.zip
ASTRA_DB_CLIENT_ID=your_client_id
ASTRA_DB_CLIENT_SECRET=your_client_secret
ASTRA_DB_KEYSPACE=your_keyspace

# Outreach Configuration
OUTREACH_API_KEY=your_outreach_api_key

# Kayako Configuration
KAYAKO_BASE_URL=https://your-kayako-instance.kayako.com
KAYAKO_EMAIL=your-email@example.com
KAYAKO_PASSWORD=your-password

# Server Configuration
HOST=0.0.0.0
PORT=8000

# LlamaCloud Configuration
LLAMACLOUD_API_KEY=your_llamacloud_api_key
LLAMACLOUD_ORG_ID=your_llamacloud_org_id
LLAMACLOUD_PROJECT_NAME=your_project_name
LLAMACLOUD_INDEX_NAME=your_index_name
```

## Testing

Run the test suite:
```bash
pytest tests/
```

For more detailed test output:
```bash
pytest tests/ -v --log-cli-level=INFO
```

## API Documentation

Once the server is running, visit:
- http://localhost:8000/docs for Swagger UI
- http://localhost:8000/redoc for ReDoc documentation

## Security Notes

- Store sensitive credentials in the `.env` file
- The `.env` file should never be committed to version control
- Consider implementing additional security measures in production

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
