"""
Node functions for the agent graph.
"""
import re
import os
import base64
import requests
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union, Tuple
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from .state import AICodeAuditState
from github import Github
from github.GithubException import GithubException
from enum import Enum
from pydantic import BaseModel, Field

# Type definitions for better type checking
StateUpdate = Dict[str, Any]
ErrorInfo = Dict[str, str]
T = TypeVar('T')

class AgentAction(str, Enum):
    """Possible actions that the agent can take."""
    DOWNLOAD_REPO_AND_MAKE_REPORT = "download_repo_and_make_report",
    ANSWER_QUESTION = "answer_question"

def check_repo_content(state: AICodeAuditState, perspective: str) -> Optional[StateUpdate]:
    """
    Check if repository content is available in the state.
    
    Args:
        state: The current state
        perspective: The name of the perspective checking the content
        
    Returns:
        StateUpdate if there's an error, None if content is available
    """
    repo_content = state.repo_content if hasattr(state, "repo_content") and state.repo_content else ""
    
    if not repo_content:
        return {
            "errors": state.errors + [{"source": f"{perspective}_perspective", "message": "Repository content not available"}],
            "messages": [AIMessage(content=f"Error: Repository content not available for {perspective} analysis. Please fetch the repository first.")]
        }
    
    return None

def call_llm_with_error_handling(
    prompt: str, 
    state: AICodeAuditState, 
    source: str, 
    success_message: str, 
    result_field: Optional[str] = None,
    model_name: str = "gpt-4o-mini"
) -> StateUpdate:
    """
    Call a language model with error handling.
    
    Args:
        prompt: The prompt to send to the model
        state: The current state
        source: The source of the call for error reporting
        success_message: The message to return on success
        result_field: The field to store the result in (if None, only messages are updated)
        model_name: The name of the model to use
        
    Returns:
        StateUpdate with the result or error information
    """
    try:
        model = ChatOpenAI(model=model_name)
        response = model.invoke(prompt)
        
        result = {"messages": [AIMessage(content=response.content)]}
        if result_field:
            result[result_field] = response.content
        
        return result
    except Exception as e:
        return {
            "errors": state.errors + [{"source": source, "message": f"Error calling language model: {str(e)}"}],
            "messages": [AIMessage(content=f"Error in {source}: {str(e)}")]
        }

def handle_perspective_analysis(
    state: AICodeAuditState,
    perspective: str,
    prompt_template: str,
    result_field: str,
    success_message: str
) -> StateUpdate:
    """
    Handle the analysis of a repository from a specific perspective.
    
    Args:
        state: The current state
        perspective: The name of the perspective
        prompt_template: The template for the prompt to send to the model
        result_field: The field to store the result in
        success_message: The message to return on success
        
    Returns:
        StateUpdate with the result or error information
    """
    try:
        # Check if repository content is available
        content_check = check_repo_content(state, perspective)
        if content_check:
            return content_check
        
        # Format the prompt with the repository content
        prompt = prompt_template.format(repo_content=state.repo_content)
        
        # Call the model with error handling
        return call_llm_with_error_handling(
            prompt=prompt,
            state=state,
            source=f"{perspective}_perspective",
            success_message=success_message,
            result_field=result_field
        )
    except Exception as e:
        return {
            "errors": state.errors + [{"source": f"{perspective}_perspective", "message": f"Unexpected error: {str(e)}"}],
            "messages": [AIMessage(content=f"An unexpected error occurred during {perspective} analysis: {str(e)}")]
        }

def starting_node(state: AICodeAuditState) -> StateUpdate:
    """
    Initial node that determines the next step based on the current state.
    
    Args:
        state: The current state
        
    Returns:
        Dictionary with updated state fields
    """
    try:
        class StartingRouterResponse(BaseModel):
            next_step: AgentAction = Field(description="The next step to take")

        # Get report if available
        report = state.report if hasattr(state, "report") and state.report else None
        
        prompt = f"""
        You are a helpful assistant that works as ai code auditor. Plan what you have to do: 
        download_repo_and_make_report - if you dont see report in the state,
        answer_question - otherwise
        
        {state.messages}
        {report}
        """
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(StartingRouterResponse)
            response = llm.invoke(prompt)
            
            # Return only updated fields
            return {"plan": response.next_step}
        except Exception as e:
            # Default to downloading repo if we can't determine the next step
            return {
                "errors": state.errors + [{"source": "starting_node", "message": f"Error calling language model: {str(e)}"}],
                "plan": AgentAction.DOWNLOAD_REPO_AND_MAKE_REPORT,
                "messages": [AIMessage(content=f"Error determining next step: {str(e)}. Defaulting to downloading repository.")]
            }
            
    except Exception as e:
        # Catch any other unexpected exceptions
        return {
            "errors": state.errors + [{"source": "starting_node", "message": f"Unexpected error: {str(e)}"}],
            "plan": AgentAction.DOWNLOAD_REPO_AND_MAKE_REPORT,
            "messages": [AIMessage(content=f"An unexpected error occurred while determining the next step: {str(e)}. Defaulting to downloading repository.")]
        }


def call_model(state: AICodeAuditState) -> StateUpdate:
    """
    Call the LLM with the current messages.
    
    Args:
        state: The current state
        config: Configuration for the graph
        
    Returns:
        Dictionary with updated state fields
    """
    try:
        # Get the messages from the state
        messages = state.messages
        
        # Get repo content and report if available
        repo_content = state.repo_content if hasattr(state, "repo_content") and state.repo_content else "No repository content available"
        report = state.report if hasattr(state, "report") and state.report else "No report available"
        
        prompt = f"""
        Here's the repo code: {repo_content}
        Here's the report: {report}
        
        Here are the messages: {messages}
        
        You're experienced ai code auditor. Always provide technical insights and recommendations - be the best possible mentor for the developer. Answer the user's question based on the repo code, prepared report and the messages:
        """
        
        return call_llm_with_error_handling(
            prompt=prompt,
            state=state,
            source="call_model",
            success_message="",  # Empty because we'll use the model's response directly
            model_name="gpt-4o-mini"
        )
            
    except Exception as e:
        # Catch any other unexpected exceptions
        return {
            "errors": state.errors + [{"source": "call_model", "message": f"Unexpected error: {str(e)}"}],
            "messages": [AIMessage(content=f"An unexpected error occurred while answering your question: {str(e)}")]
        }


def fetch_github_repo_node(state: AICodeAuditState) -> StateUpdate:
    """
    Fetches a GitHub repository, extracts its structure and file contents.
    
    Args:
        state: The current state containing repository URL and GitHub token
        
    Returns:
        Dictionary with updated state fields
    """
    try:
        # Extract GitHub token and repository URL from state
        github_token = os.environ.get("GITHUB_TOKEN")
        repo_url = state.repo_url if hasattr(state, "repo_url") and state.repo_url else ""
        
        if not github_token:
            return {
                "errors": state.errors + [{"source": "fetch_github_repo", "message": "GitHub token not provided in environment variables"}],
                "messages": [AIMessage(content="Error: GitHub token not provided. Please check your environment variables.")]
            }
        
        if not repo_url:
            return {
                "errors": state.errors + [{"source": "fetch_github_repo", "message": "Repository URL not provided in state"}],
                "messages": [AIMessage(content="Error: Repository URL not provided. Please provide a valid GitHub repository URL.")]
            }
        
        # Extract repo owner and name from URL
        # Format: https://github.com/owner/repo
        match = re.search(r"github.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            return {
                "errors": state.errors + [{"source": "fetch_github_repo", "message": f"Invalid GitHub repository URL: {repo_url}"}],
                "messages": [AIMessage(content=f"Error: Invalid GitHub repository URL: {repo_url}. Please provide a valid URL in the format https://github.com/owner/repo.")]
            }
        
        owner, repo_name = match.groups()
        
        # Initialize GitHub client
        g = Github(github_token)
        
        try:
            repo = g.get_repo(f"{owner}/{repo_name}")
        except GithubException as e:
            return {
                "errors": state.errors + [{"source": "fetch_github_repo", "message": f"GitHub API error: {str(e)}"}],
                "messages": [AIMessage(content=f"Error accessing repository: {str(e)}. Please check the repository URL and your GitHub token permissions.")]
            }
        
        # Get repository structure
        try:
            contents = repo.get_contents("")
            repo_structure = []
            file_contents = []
            
            # Process repository contents recursively
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    # Add directory to structure
                    repo_structure.append(f"📁 {file_content.path}")
                    # Get contents of this directory
                    contents.extend(repo.get_contents(file_content.path))
                else:
                    # Add file to structure
                    repo_structure.append(f"📄 {file_content.path}")
                    
                    # Get file content
                    try:
                        # Skip large files and binary files
                        if file_content.size > 1000000 or _is_binary_path(file_content.path):
                            file_data = f"\n{'='*80}\nFILE: {file_content.path}\n{'='*80}\n[File too large or binary, content skipped]\n"
                        else:
                            content = base64.b64decode(file_content.content).decode('utf-8', errors='replace')
                            file_data = f"\n{'='*80}\nFILE: {file_content.path}\n{'='*80}\n{content}\n"
                        
                        file_contents.append(file_data)
                    except (UnicodeDecodeError, AttributeError):
                        file_data = f"\n{'='*80}\nFILE: {file_content.path}\n{'='*80}\n[Could not decode file content]\n"
                        file_contents.append(file_data)
            
            # Combine structure and contents
            structure_str = "REPOSITORY STRUCTURE:\n" + "\n".join(repo_structure)
            content_str = "\n\nFILE CONTENTS:" + "".join(file_contents)
            
            repo_data = structure_str + content_str
            
            # Return only updated fields
            return {
                "messages": [AIMessage(content=f"Successfully fetched repository: {owner}/{repo_name}")],
                "repo_content": repo_data
            }
        except GithubException as e:
            return {
                "errors": state.errors + [{"source": "fetch_github_repo", "message": f"Error fetching repository contents: {str(e)}"}],
                "messages": [AIMessage(content=f"Error fetching repository contents: {str(e)}. The repository might be empty or inaccessible.")]
            }
            
    except Exception as e:
        # Catch any other unexpected exceptions
        return {
            "errors": state.errors + [{"source": "fetch_github_repo", "message": f"Unexpected error: {str(e)}"}],
            "messages": [AIMessage(content=f"An unexpected error occurred while fetching the repository: {str(e)}")]
        }

def _is_binary_path(path):
    """Helper function to identify likely binary files by extension"""
    binary_extensions = [
        '.exe', '.bin', '.o', '.so', '.dll', '.obj', '.png', '.jpg', 
        '.jpeg', '.gif', '.bmp', '.ico', '.pdf', '.zip', '.tar', 
        '.gz', '.rar', '.7z', '.pyc', '.class', '.jar'
    ]
    return any(path.lower().endswith(ext) for ext in binary_extensions)


def architectural_perspective_node(state: AICodeAuditState) -> StateUpdate:
    """
    Analyzes repository from an architectural perspective.
    
    Args:
        state: The current state containing repository content
        
    Returns:
        Dictionary with updated state fields
    """
    prompt_template = """
    # Architectural Code Review

    Below is the code repository content. Please analyze it from an ARCHITECTURAL perspective.

    Focus on:
    - Overall architecture patterns and design
    - Component organization and modularity
    - Dependencies between modules
    - Architectural strengths and weaknesses
    - Scalability concerns
    - Potential architectural improvements

    Format your response as a professional Markdown report with the following sections:
    1. Executive Summary
    2. Architecture Overview (with diagrams descriptions if possible)
    3. Component Analysis
    4. Dependencies Assessment
    5. Architectural Strengths
    6. Architectural Weaknesses
    7. Scalability Evaluation
    8. Recommendations for Improvement

    REPOSITORY CONTENT:
    {repo_content}
    """
    
    return handle_perspective_analysis(
        state=state,
        perspective="architectural",
        prompt_template=prompt_template,
        result_field="architectural_report",
        success_message="Architectural perspective report generated successfully"
    )


def business_domain_perspective_node(state: AICodeAuditState) -> StateUpdate:
    """
    Analyzes repository from a business/domain perspective.
    
    Args:
        state: The current state containing repository content
        
    Returns:
        Dictionary with updated state fields
    """
    prompt_template = """
    # Business/Domain Code Review

    Below is the code repository content. Please analyze it from a BUSINESS/DOMAIN perspective.

    Focus on:
    - Alignment with business requirements
    - Domain model clarity
    - Business rules implementation
    - Domain language usage in code
    - Coverage of business functions
    - Business value delivery
    - Missing business functionality

    Format your response as a professional Markdown report with the following sections:
    1. Executive Summary
    2. Domain Model Assessment
    3. Business Logic Evaluation
    4. Domain Language Analysis
    5. Functional Coverage
    6. Business Value Alignment
    7. Gaps and Missing Functionality
    8. Recommendations for Business Alignment

    REPOSITORY CONTENT:
    {repo_content}
    """
    
    return handle_perspective_analysis(
        state=state,
        perspective="business_domain",
        prompt_template=prompt_template,
        result_field="business_domain_report",
        success_message="Business/Domain perspective report generated successfully"
    )


def code_quality_perspective_node(state: AICodeAuditState) -> StateUpdate:
    """
    Analyzes repository from a code quality perspective.
    
    Args:
        state: The current state containing repository content
        
    Returns:
        Dictionary with updated state fields
    """
    prompt_template = """
    # Code Quality Review

    Below is the code repository content. Please analyze it from a CODE QUALITY perspective.

    Focus on:
    - Clean code principles (SOLID, DRY, KISS)
    - Code smells and anti-patterns
    - Code complexity and readability
    - Naming conventions and consistency
    - Error handling and edge cases
    - Testing quality and coverage
    - Documentation and comments

    Format your response as a professional Markdown report with the following sections:
    1. Executive Summary
    2. Clean Code Assessment
    3. Code Smells and Anti-patterns
    4. Complexity Analysis
    5. Readability Evaluation
    6. Error Handling Review
    7. Testing Evaluation
    8. Documentation Assessment
    9. Recommendations for Improvement

    REPOSITORY CONTENT:
    {repo_content}
    """
    
    return handle_perspective_analysis(
        state=state,
        perspective="code_quality",
        prompt_template=prompt_template,
        result_field="code_quality_report",
        success_message="Code quality perspective report generated successfully"
    )


def security_perspective_node(state: AICodeAuditState) -> StateUpdate:
    """
    Analyzes repository from a security perspective.
    
    Args:
        state: The current state containing repository content
        
    Returns:
        Dictionary with updated state fields
    """
    prompt_template = """
    # Security Code Review

    Below is the code repository content. Please analyze it from a SECURITY perspective.

    Focus on:
    - Common security vulnerabilities (OWASP Top 10 if applicable)
    - Authentication and authorization mechanisms
    - Data validation and sanitization
    - Secure coding practices
    - Sensitive data handling
    - Cryptography usage
    - Security configurations

    Format your response as a professional Markdown report with the following sections:
    1. Executive Summary
    2. Vulnerability Assessment
    3. Authentication & Authorization Review
    4. Data Validation Analysis
    5. Sensitive Data Handling
    6. Cryptography Evaluation
    7. Security Configuration Review
    8. Risk Assessment
    9. Security Recommendations

    REPOSITORY CONTENT:
    {repo_content}
    """
    
    return handle_perspective_analysis(
        state=state,
        perspective="security",
        prompt_template=prompt_template,
        result_field="security_report",
        success_message="Security perspective report generated successfully"
    )


def modernization_perspective_node(state: AICodeAuditState) -> StateUpdate:
    """
    Analyzes repository from a modernization perspective.
    
    Args:
        state: The current state containing repository content
        
    Returns:
        Dictionary with updated state fields
    """
    prompt_template = """
    # Modernization Code Review

    Below is the code repository content. Please analyze it from a MODERNIZATION perspective.

    Focus on:
    - Outdated technologies, frameworks, and libraries
    - Legacy code patterns and approaches
    - Technical debt
    - Modern alternatives for current implementations
    - Opportunities for adopting new technologies
    - Migration paths and strategies
    - Potential performance improvements with modern approaches

    Format your response as a professional Markdown report with the following sections:
    1. Executive Summary
    2. Technology Stack Assessment
    3. Legacy Code Evaluation
    4. Technical Debt Analysis
    5. Modernization Opportunities
    6. Suggested Technologies and Approaches
    7. Migration Strategy
    8. Performance Improvement Potential

    REPOSITORY CONTENT:
    {repo_content}
    """
    
    return handle_perspective_analysis(
        state=state,
        perspective="modernization",
        prompt_template=prompt_template,
        result_field="modernization_report",
        success_message="Modernization perspective report generated successfully"
    )

def generate_report_node(state: AICodeAuditState) -> StateUpdate:
    """
    Generates a comprehensive report from all perspectives.
    
    Args:
        state: The current state containing all perspectives
        
    Returns:
        Dictionary with updated state fields
    """
    try:
        # Check if all perspective reports are available
        missing_reports = []
        if not hasattr(state, "architectural_report") or not state.architectural_report:
            missing_reports.append("architectural")
        if not hasattr(state, "business_domain_report") or not state.business_domain_report:
            missing_reports.append("business/domain")
        if not hasattr(state, "code_quality_report") or not state.code_quality_report:
            missing_reports.append("code quality")
        if not hasattr(state, "security_report") or not state.security_report:
            missing_reports.append("security")
        if not hasattr(state, "modernization_report") or not state.modernization_report:
            missing_reports.append("modernization")
        
        if missing_reports:
            return {
                "errors": state.errors + [{"source": "generate_report", "message": f"Missing perspective reports: {', '.join(missing_reports)}"}],
                "messages": [AIMessage(content=f"Error: Cannot generate comprehensive report. Missing perspective reports: {', '.join(missing_reports)}")]
            }
        
        prompt = f"""
        # Comprehensive Code Audit Report

        Below are the perspectives on the code repository. Please generate a comprehensive report by combining all the perspectives.

        ARCHITECTURAL PERSPECTIVE:  
        {state.architectural_report}

        BUSINESS/DOMAIN PERSPECTIVE:
        {state.business_domain_report}

        CODE QUALITY PERSPECTIVE:
        {state.code_quality_report}    

        SECURITY PERSPECTIVE:
        {state.security_report}

        MODERNIZATION PERSPECTIVE:
        {state.modernization_report}     
        
        Please generate a comprehensive report by combining all the perspectives.
        """
        
        return call_llm_with_error_handling(
            prompt=prompt,
            state=state,
            source="generate_report",
            success_message="Comprehensive report generated successfully",
            result_field="report",
            model_name="gpt-4o-mini"
        )
            
    except Exception as e:
        # Catch any other unexpected exceptions
        return {
            "errors": state.errors + [{"source": "generate_report", "message": f"Unexpected error: {str(e)}"}],
            "messages": [AIMessage(content=f"An unexpected error occurred while generating the comprehensive report: {str(e)}")]
        }