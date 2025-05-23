"""
PTR Knowledge Base Agent Component
This component provides an agent that can answer questions about PTR using Astra DB as the knowledge base.
"""

from .agent import PTRKnowledgeAgent
from .models import PTRKnowledgeConfig

__all__ = ["PTRKnowledgeAgent", "PTRKnowledgeConfig"] 