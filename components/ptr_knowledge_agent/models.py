"""
Configuration models for the PTR Knowledge Base Agent.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List

class PTRKnowledgeConfig(BaseModel):
    """Configuration for the PTR Knowledge Base Agent."""
    
    # Agent Configuration
    model_name: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    temperature: float = Field(default=0.7, description="Temperature for response generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens for response")
    
    # Optional Configuration
    system_message: Optional[str] = Field(
        default=None,
        description="Custom system message for the agent"
    )
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError('Temperature must be between 0 and 1')
        return v
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max_tokens is positive."""
        if v <= 0:
            raise ValueError('max_tokens must be positive')
        return v
    
    model_config = ConfigDict(arbitrary_types_allowed=True) 