"""
State definitions for the agent.
"""
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing import List, Annotated, Dict, Optional

class AICodeAuditState(BaseModel):
    """State for the agent workflow."""
   
    # The messages passed between the agent and user
    messages: Annotated[List[AnyMessage], add_messages]
    plan: str = Field(description="Plan of the agent", default=None)
    
    # Error tracking
    errors: List[Dict[str, str]] = Field(
        description="List of errors encountered during processing",
        default_factory=list
    )
    
    # GitHub repository data
    repo_url: str = Field(description="URL of the GitHub repository to audit", default=None)
    repo_content: List[str] = Field(description="Content chunks of the GitHub repository", default_factory=list)
    repo_structure: str = Field(description="Structure of the GitHub repository", default=None)
    
    # Analysis reports
    architectural_report: str = Field(description="Architectural report of the GitHub repository", default=None)
    business_domain_report: str = Field(description="Business/Domain report of the GitHub repository", default=None)
    code_quality_report: str = Field(description="Code quality report of the GitHub repository", default=None)
    security_report: str = Field(description="Security report of the GitHub repository", default=None)
    modernization_report: str = Field(description="Modernization report of the GitHub repository", default=None) 
    report: str = Field(description="Comprehensive report of the GitHub repository", default=None)
    
    # Partial reports (for chunked analysis)
    partial_architectural_reports: List[str] = Field(description="Partial architectural reports for each chunk", default_factory=list)
    partial_business_domain_reports: List[str] = Field(description="Partial business/domain reports for each chunk", default_factory=list)
    partial_code_quality_reports: List[str] = Field(description="Partial code quality reports for each chunk", default_factory=list)
    partial_security_reports: List[str] = Field(description="Partial security reports for each chunk", default_factory=list)
    partial_modernization_reports: List[str] = Field(description="Partial modernization reports for each chunk", default_factory=list)
    