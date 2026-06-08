"""LangGraph compilation and conditional edge routing."""

import logging
from langgraph.graph import StateGraph, START, END
from src.models.state import LeadState
from src.nodes.discovery import discovery_node
from src.nodes.web_crawler import web_crawler_node
from src.nodes.consensus import consensus_node
from src.nodes.enrichment import enrichment_node

logger = logging.getLogger(__name__)


def route_after_discovery(state: LeadState) -> str:
    if state.get("direct_to_enrichment", False):
        return "enrichment"
    return "web_crawler"


def should_continue_after_consensus(state: LeadState) -> str:
    """
    Conditional edge after Node 3 (Consensus).

    Routes to Node 4 if lead passed consensus (score >= threshold),
    otherwise to END.
    """
    passed = state.get("consensus_passed", False)

    if passed:
        logger.info("Consensus → Node 4 (Enrichment)")
        return "enrichment"
    else:
        logger.info("Consensus → END")
        return "end"


def build_graph() -> StateGraph:
    """
    Build and compile the complete LangGraph pipeline.

    Graph structure:
        START(main.py)
          ↓
        Node 1: Discovery (Registry Check)
          ├→ Active/Unknown/Not Found → Node 2: Web Crawler (Signal Extraction)
          │             ↓
          │           Node 3: Consensus (Deterministic Scoring)
          │             ├→ Passed → Node 4: Enrichment (Hunter.io)
          │             │             ↓
          │             │           END (Return Lead)
          │             │
          │             └→ Failed → END
          │
          └→ Inactive/Suspended/Dissolved → END

    Returns:
        Compiled StateGraph ready for ainvoke()
    """
    workflow = StateGraph(LeadState)

    # Add nodes
    workflow.add_node("discovery", discovery_node)
    workflow.add_node("web_crawler", web_crawler_node)
    workflow.add_node("consensus", consensus_node)
    workflow.add_node("enrichment", enrichment_node)

    # Set entry point
    workflow.set_entry_point("discovery")

    # Add conditional edges
    workflow.add_conditional_edges(
        "discovery",
        route_after_discovery,
        {
            "enrichment": "enrichment",
            "web_crawler": "web_crawler",
        },
    )

    # Sequential edges (always proceed to next node)
    workflow.add_edge("web_crawler", "consensus")

    workflow.add_conditional_edges(
        "consensus",
        should_continue_after_consensus,
        {
            "enrichment": "enrichment",
            "end": END,
        },
    )

    workflow.add_edge("enrichment", END)

    # Compile the graph
    compiled_graph = workflow.compile()
    logger.info("LangGraph compiled successfully")

    return compiled_graph
