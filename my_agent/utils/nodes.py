"""
Node functions for the agent graph.
"""
import re
import os
import base64
import requests
from typing import Dict
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from .state import AICodeAuditState
from github import Github
from github.GithubException import GithubException
from enum import Enum
from pydantic import BaseModel, Field

class NextStep(str, Enum):
    DOWNLOAD_REPO_AND_MAKE_REPORT = "download_repo_and_make_report",
    ANSWER_QUESTION = "answer_question"

def starting_node(state: AICodeAuditState) -> AICodeAuditState:
    
    class StartingRouterResponse(BaseModel):
        next_step:  NextStep = Field(description="The next step to take")

    prompt = f"""
    You are a helpful assistant that works as ai code auditor. Plan what you have to do: 
    download_repo_and_make_report - if you dont see report in the state,
    answer_question - otherwise
    
    {state.messages}
    {state.report}
    """
    llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(StartingRouterResponse)
    
    response = llm.invoke(prompt)
    
    return {"plan": response.next_step}


    

def call_model(state: AICodeAuditState) -> AICodeAuditState:
    """
    Call the LLM with the current messages.
    
    Args:
        state: The current state
        config: Configuration for the graph
        
    Returns:
        Updated state with the model's response
    """
    model = ChatOpenAI(model="gpt-4o-mini")
      # Get the messages from the state
    messages = state.messages
    
    prompt = f"""
    Here's the repo code: {state.repo_content}
    Here's the report: {state.report}
    
    Here are the messages: {messages}
    
    Answer the user's question based on the repo code, prepared report and the messages:
    """
    
    # Call the model
    response = model.invoke(prompt)
    
    return {"messages": [response]}


def fetch_github_repo_node(state: AICodeAuditState) -> AICodeAuditState:
    """
    Fetches a GitHub repository, extracts its structure and file contents.
    
    Args:
        state: The current state containing repository URL and GitHub token
        
    Returns:
        Updated state with repository structure and file contents as a string
    """
    # Extract GitHub token and repository URL from state
    print('hello! 1')
    github_token = os.environ.get("GITHUB_TOKEN")
    repo_url = state.repo_url if hasattr(state, "repo_url") else ""
    
    if not github_token:
        return {"error": "GitHub token not provided in state or environment variables"}
    
    if not repo_url:
        return {"error": "Repository URL not provided in state"}
    
    # Extract repo owner and name from URL
    # Format: https://github.com/owner/repo
    match = re.search(r"github.com/([^/]+)/([^/]+)", repo_url)
    if not match:
        return {"error": f"Invalid GitHub repository URL: {repo_url}"}
    
    owner, repo_name = match.groups()
    
    
    # Initialize GitHub client
    g = Github(github_token)
    repo = g.get_repo(f"{owner}/{repo_name}")
    
    # Get repository structure
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
    
    # Update and return state
    
    print('hello!')
    
    return {
        "messages": [AIMessage(content=f"Successfully fetched repository: {owner}/{repo_name}")],
        "repo_content": repo_data,
    }

def _is_binary_path(path):
    """Helper function to identify likely binary files by extension"""
    binary_extensions = [
        '.exe', '.bin', '.o', '.so', '.dll', '.obj', '.png', '.jpg', 
        '.jpeg', '.gif', '.bmp', '.ico', '.pdf', '.zip', '.tar', 
        '.gz', '.rar', '.7z', '.pyc', '.class', '.jar'
    ]
    return any(path.lower().endswith(ext) for ext in binary_extensions)


def architectural_perspective_node(state: AICodeAuditState) -> AICodeAuditState:
    """
    Analyzes repository from an architectural perspective.
    
    Args:
        state: The current state containing repository content
        
    Returns:
        Updated state with architectural perspective report
    """
    # Get the repo content from the state
    repo_content = state.repo_content if hasattr(state, "repo_content") else ""
    
    if not repo_content:
        return {"error": "Repository content not available"}
    
    model = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = f"""
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
    
    # Call the model
    response = model.invoke(prompt)
    
    # Return updated state
    return {
        "messages": [AIMessage(content="Architectural perspective report generated successfully")],
        "architectural_report": response.content
    }


def business_domain_perspective_node(state: AICodeAuditState) -> AICodeAuditState:
    """
    Analyzes repository from a business/domain perspective.
    
    Args:
        state: The current state containing repository content
        
    Returns:
        Updated state with business/domain perspective report
    """
    # Get the repo content from the state
    repo_content = state.repo_content if hasattr(state, "repo_content") else ""
    
    if not repo_content:
        return {"error": "Repository content not available"}
    
    model = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = f"""
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
    
    # Call the model
    response = model.invoke(prompt)
    
    # Return updated state
    return {
        "messages": [AIMessage(content="Business/Domain perspective report generated successfully")],
        "business_domain_report": response.content
    }


def code_quality_perspective_node(state: AICodeAuditState) -> AICodeAuditState:
    """
    Analyzes repository from a code quality perspective.
    
    Args:
        state: The current state containing repository content
        
    Returns:
        Updated state with code quality perspective report
    """
    # Get the repo content from the state
    repo_content = state.repo_content if hasattr(state, "repo_content") else ""
    
    if not repo_content:
        return {"error": "Repository content not available"}
    
    model = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = f"""
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
    7. Testing Quality
    8. Documentation Review
    9. Recommendations for Quality Improvement

    REPOSITORY CONTENT:
    {repo_content}
    """
    
    # Call the model
    response = model.invoke(prompt)
    
    # Return updated state
    return {
        "messages": [AIMessage(content="Code quality perspective report generated successfully")],
        "code_quality_report": response.content
    }


def security_perspective_node(state: AICodeAuditState) -> AICodeAuditState:
    """
    Analyzes repository from a security perspective.
    
    Args:
        state: The current state containing repository content
        
    Returns:
        Updated state with security perspective report
    """
    # Get the repo content from the state
    repo_content = state.repo_content if hasattr(state, "repo_content") else ""
    
    if not repo_content:
        return {"error": "Repository content not available"}
    
    model = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = f"""
    # Security Code Review

    Below is the code repository content. Please analyze it from a SECURITY perspective.

    Focus on:
    - Potential security vulnerabilities (OWASP Top 10)
    - Input validation and sanitization
    - Authentication and authorization mechanisms
    - Sensitive data handling
    - Cryptographic practices
    - Secure configuration
    - Third-party dependencies security
    - Security best practices compliance

    Format your response as a professional Markdown report with the following sections:
    1. Executive Summary
    2. Vulnerability Assessment
    3. Authentication & Authorization Review
    4. Data Security Analysis
    5. Dependency Security Evaluation
    6. Security Best Practices Compliance
    7. Risk Assessment
    8. Recommendations for Security Hardening

    REPOSITORY CONTENT:
    {repo_content}
    """
    
    # Call the model
    response = model.invoke(prompt)
    
    # Return updated state
    return {
        "messages": [AIMessage(content="Security perspective report generated successfully")],
        "security_report": response.content
    }


def modernization_perspective_node(state: AICodeAuditState) -> AICodeAuditState:
    """
    Analyzes repository from a modernization perspective.
    
    Args:
        state: The current state containing repository content
        
    Returns:
        Updated state with modernization perspective report
    """
    # Get the repo content from the state
    repo_content = state.repo_content if hasattr(state, "repo_content") else ""
    
    if not repo_content:
        return {"error": "Repository content not available"}
    
    model = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = f"""
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
    8. Cost-Benefit Analysis of Modernization

    REPOSITORY CONTENT:
    {repo_content}
    """
    
    # Call the model
    response = model.invoke(prompt)
    
    # Return updated state
    return {
        "messages": [AIMessage(content='Modernization perspective report generated successfully')],
        "modernization_report": response.content
    }    

def generate_report_node(state: AICodeAuditState) -> AICodeAuditState:
    """
    Generates a comprehensive report from all perspectives.
    
    Args:
        state: The current state containing all perspectives
        
    Returns:
        Updated state with comprehensive report
    """
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
    
    model = ChatOpenAI(model="gpt-4o-mini")
    # Call the model
    response = model.invoke(prompt)
    
    # Return updated state
    return { 
        "messages": [AIMessage(content="Comprehensive report generated successfully")],
        "report": response.content
    }