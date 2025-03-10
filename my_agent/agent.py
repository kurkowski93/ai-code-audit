"""
Agent graph implementation.
"""
from langgraph.graph import StateGraph, END, START
from my_agent.utils.nodes import call_model, fetch_github_repo_node, architectural_perspective_node, business_domain_perspective_node, code_quality_perspective_node, security_perspective_node, modernization_perspective_node, generate_report_node, starting_router, starting_node, NextStep
from my_agent.utils.state import AICodeAuditState



# Create the graph
workflow = StateGraph(AICodeAuditState)

# Add nodes
workflow.add_node("starting_node", starting_node)
workflow.add_node("answer_question", call_model)
workflow.add_node("fetch_repo", fetch_github_repo_node)
workflow.add_node("architectural_perspective", architectural_perspective_node)
workflow.add_node("business_domain_perspective", business_domain_perspective_node)
workflow.add_node("code_quality_perspective", code_quality_perspective_node)
workflow.add_node("security_perspective", security_perspective_node)
workflow.add_node("modernization_perspective", modernization_perspective_node)
workflow.add_node("generate_report", generate_report_node)

def starting_router(state: AICodeAuditState) -> str:
    if state.plan == NextStep.DOWNLOAD_REPO_AND_MAKE_REPORT:
        return "fetch_repo"
    elif state.plan == NextStep.ANSWER_QUESTION:
        return "answer_question"


# Define the flow
workflow.add_edge(START, "starting_node")
workflow.add_conditional_edges("starting_node", starting_router, {"fetch_repo": "fetch_repo", "answer_question": "answer_question"})

workflow.add_edge("fetch_repo", "architectural_perspective")
workflow.add_edge("fetch_repo", "business_domain_perspective")
workflow.add_edge("fetch_repo", "code_quality_perspective")
workflow.add_edge("fetch_repo", "security_perspective")
workflow.add_edge("fetch_repo", "modernization_perspective")

workflow.add_edge("architectural_perspective", "generate_report")
workflow.add_edge("business_domain_perspective", "generate_report")
workflow.add_edge("code_quality_perspective", "generate_report")
workflow.add_edge("security_perspective", "generate_report")
workflow.add_edge("modernization_perspective", "generate_report")
workflow.add_edge("generate_report", END)

workflow.add_edge("answer_question", END)


# Compile the graph
graph = workflow.compile() 