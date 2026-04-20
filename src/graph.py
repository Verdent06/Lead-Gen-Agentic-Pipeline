"""LangGraph compilation and conditional edge routing."""

import logging
from langgraph.graph import StateGraph, START, END
from src.models.state import LeadState
from src.nodes.discovery import discovery_node
from src.nodes.web_crawler import web_crawler_node
from src.nodes.consensus import consensus_node
from src.nodes.enrichment import enrichment_node

logger = logging.getLogger(__name__)


def should_continue_after_discovery(state: LeadState) -> str:
    """
    Conditional edge after Node 1 (Discovery).

    Routes to Node 2 if registry status is 'active', 'unknown', or 'not_found'.
    Routes to END only if status is 'inactive', 'suspended', or 'dissolved'.
    """
    status = state.get("registry_verification_status", "")
    should_continue = state.get("should_continue", False)

    if should_continue:
        logger.info(f"Discovery → Node 2 (Web Crawler) [status: {status}]")
        return "web_crawler"
    else:
        logger.info(f"Discovery → END (status: {status})")
        return "end"


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
        should_continue_after_discovery,
        {
            "web_crawler": "web_crawler",
            "end": END,
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
