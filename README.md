# AI Code Audit

A comprehensive code audit agent built with LangGraph and LangChain that analyzes GitHub repositories from multiple perspectives and generates detailed reports.

## Features

- **Multi-perspective Analysis**: Analyzes repositories from architectural, business/domain, code quality, security, and modernization perspectives
- **Comprehensive Reports**: Generates detailed reports for each perspective and a combined comprehensive report
- **Error Handling**: Robust error handling with detailed error messages
- **GitHub Integration**: Fetches repositories directly from GitHub for analysis
- **Interactive Q&A**: Ask questions about the analyzed repository and get AI-powered answers

## Architecture

The agent is built using LangGraph, which provides a graph-based workflow for the analysis process:

1. **Starting Node**: Determines whether to download a repository and generate a report or answer a question
2. **Repository Fetching**: Downloads and processes a GitHub repository
3. **Perspective Analysis**: Analyzes the repository from multiple perspectives in parallel
4. **Report Generation**: Combines the perspective reports into a comprehensive report
5. **Question Answering**: Answers questions about the repository based on the generated reports

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
GITHUB_TOKEN=your_github_token
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

### Running the Agent

To run the agent using the LangGraph CLI:

```bash
langgraph run agent
```

### Analyzing a Repository

1. Start the agent
2. Provide a GitHub repository URL (e.g., https://github.com/username/repository)
3. Wait for the analysis to complete
4. Review the generated reports

### Asking Questions

After a repository has been analyzed, you can ask questions about it:

1. Ask a specific question about the repository
2. The agent will use the generated reports to provide a detailed answer

## Project Structure

- `my_agent/`: Main agent code
  - `agent.py`: LangGraph workflow definition
  - `utils/`: Utility functions and components
    - `nodes.py`: Node functions for the graph
    - `state.py`: State definitions
    - `tools.py`: Tools for the agent

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 