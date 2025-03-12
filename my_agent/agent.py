"""
Agent graph implementation.
"""
from langgraph.graph import StateGraph, END, START
from my_agent.utils.nodes import answer_question, fetch_repo_node, architectural_perspective_node, business_domain_perspective_node, code_quality_perspective_node, security_perspective_node, modernization_perspective_node, generate_report_node, starting_node, AgentAction
from my_agent.utils.state import AICodeAuditState



# Create the graph
code_audit_graph = StateGraph(AICodeAuditState)

# Add nodes
code_audit_graph.add_node("starting_node", starting_node)
code_audit_graph.add_node("answer_question", answer_question)
code_audit_graph.add_node("repository_scanner", fetch_repo_node)
code_audit_graph.add_node("architectural_perspective", architectural_perspective_node)
code_audit_graph.add_node("business_domain_perspective", business_domain_perspective_node)
code_audit_graph.add_node("code_quality_perspective", code_quality_perspective_node)
code_audit_graph.add_node("security_perspective", security_perspective_node)
code_audit_graph.add_node("modernization_perspective", modernization_perspective_node)
code_audit_graph.add_node("generate_report", generate_report_node)

def determine_next_action(state: AICodeAuditState) -> str:
    if state.plan == AgentAction.DOWNLOAD_REPO_AND_MAKE_REPORT:
        return "repository_scanner"
    elif state.plan == AgentAction.ANSWER_QUESTION:
        return "answer_question"


# Define the flow
code_audit_graph.add_edge(START, "starting_node")
code_audit_graph.add_conditional_edges("starting_node", determine_next_action, {"repository_scanner": "repository_scanner", "answer_question": "answer_question"})

code_audit_graph.add_edge("repository_scanner", "architectural_perspective")
code_audit_graph.add_edge("repository_scanner", "business_domain_perspective")
code_audit_graph.add_edge("repository_scanner", "code_quality_perspective")
code_audit_graph.add_edge("repository_scanner", "security_perspective")
code_audit_graph.add_edge("repository_scanner", "modernization_perspective")

code_audit_graph.add_edge("architectural_perspective", "generate_report")
code_audit_graph.add_edge("business_domain_perspective", "generate_report")
code_audit_graph.add_edge("code_quality_perspective", "generate_report")
code_audit_graph.add_edge("security_perspective", "generate_report")
code_audit_graph.add_edge("modernization_perspective", "generate_report")
code_audit_graph.add_edge("generate_report", END)

code_audit_graph.add_edge("answer_question", END)


# Compile the graph
graph = code_audit_graph.compile() 