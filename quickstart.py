#!/usr/bin/env python3
"""
Quick Start Guide - Autonomous Sourcing Agent

Run this to verify installation and test with mock data.
"""

import sys
import asyncio
import os


async def quick_start():
    """Run quick start verification."""
    print("\n" + "=" * 70)
    print("AUTONOMOUS SOURCING AGENT - QUICK START")
    print("=" * 70)

    # Step 1: Check environment
    print("\n[1] Checking environment...")
    try:
        import langgraph
        import pydantic
        import langchain_google_genai
        import httpx
        print("    ✓ Core dependencies installed")
    except ImportError as e:
        print(f"    ✗ Missing dependency: {e}")
        print("      Run: pip install -r requirements.txt")
        return False

    # Step 2: Validate config
    print("\n[2] Loading configuration...")
    try:
        from src.config import Config
        Config.validate()
        print("    ✓ Configuration valid")
        print(f"    - USE_MOCKS: {Config.USE_MOCKS}")
        print(f"    - LLM_MODEL: {Config.LLM_MODEL}")
    except Exception as e:
        print(f"    ✗ Configuration error: {e}")
        return False

    # Step 3: Import models
    print("\n[3] Importing models...")
    try:
        from src.models.state import LeadState
        from src.models.schemas import (
            RegistryVerification,
            WebsiteSignals,
            ConsensusResult,
            HunterContact,
            FinalLeadOutput,
        )
        print("    ✓ All models imported successfully")
    except Exception as e:
        print(f"    ✗ Model import error: {e}")
        return False

    # Step 4: Build graph
    print("\n[4] Building LangGraph...")
    try:
        from src.graph import build_graph
        graph = build_graph()
        print("    ✓ Graph compiled successfully")
    except Exception as e:
        print(f"    ✗ Graph build error: {e}")
        return False

    # Step 5: Run pipeline with mock data
    print("\n[5] Running pipeline with mock data...")
    print("    (This uses hardcoded mock responses, no real API calls)")

    try:
        from src.main import run_sourcing_agent

        # Force mock mode
        os.environ["USE_MOCKS"] = "true"

        result = await run_sourcing_agent(
            query="Find HVAC distributors in Ohio without e-commerce",
            business_name="Example Company Corp",
            location="Cleveland, Ohio",
        )

        print("\n    ✓ Pipeline executed successfully!")
        print(f"\n    Results:")
        print(f"      - Business: {result.business_name}")
        print(f"      - Lead Score: {result.lead_score}/100")
        print(f"      - Status: {'PASSED' if result.passed_consensus else 'REJECTED'}")
        print(f"      - Execution Time: {result.execution_time_seconds:.2f}s")

        if result.primary_contact:
            print(f"\n    Primary Contact:")
            print(f"      - Name: {result.primary_contact.first_name} {result.primary_contact.last_name}")
            print(f"      - Email: {result.primary_contact.email}")
            print(f"      - Title: {result.primary_contact.job_title}")

        print(f"\n    Recommendation: {result.recommendation}")

        return True

    except Exception as e:
        print(f"    ✗ Pipeline error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main entry point."""
    success = await quick_start()

    print("\n" + "=" * 70)
    if success:
        print("✓ QUICK START COMPLETED SUCCESSFULLY")
        print("\nNext steps:")
        print("  1. Set up .env with real API keys:")
        print("     - GOOGLE_API_KEY (https://makersuite.google.com/app/apikey)")
        print("     - TAVILY_API_KEY (https://tavily.com)")
        print("     - HUNTER_API_KEY (https://hunter.io)")
        print("\n  2. Run with real APIs:")
        print("     export USE_MOCKS=false")
        print("     python -m src.main")
        print("\n  3. Customize for your use case:")
        print("     - Adjust SIGNAL_SCORES in src/nodes/consensus.py")
        print("     - Update signal extraction prompts in src/nodes/web_crawler.py")
        print("     - Add custom signals to WebsiteSignals in src/models/schemas.py")
    else:
        print("✗ QUICK START FAILED - See errors above")
        sys.exit(1)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
