#!/usr/bin/env bash
# Verification script to confirm all async patterns are correct

echo "=========================================="
echo "ASYNC ARCHITECTURE VERIFICATION"
echo "=========================================="
echo ""

# Check 1: Verify graph.ainvoke is used (not asyncio.to_thread)
echo "[1] Checking for correct ainvoke() usage..."
if grep -q "await graph.ainvoke(initial_state)" /Users/vedantdesai/Projects/Lead-Gen\ Agentic\ Pipeline/src/main.py; then
    echo "✓ PASS: Using native ainvoke()"
else
    echo "✗ FAIL: Not using ainvoke()"
fi

if ! grep -q "asyncio.to_thread" /Users/vedantdesai/Projects/Lead-Gen\ Agentic\ Pipeline/src/main.py; then
    echo "✓ PASS: No asyncio.to_thread() wrapper"
else
    echo "✗ FAIL: Still using asyncio.to_thread()"
fi
echo ""

# Check 2: Verify all nodes are async def
echo "[2] Checking node definitions..."
for node in discovery web_crawler consensus enrichment; do
    if grep -q "async def ${node}_node" /Users/vedantdesai/Projects/Lead-Gen\ Agentic\ Pipeline/src/nodes/${node}.py; then
        echo "✓ PASS: ${node}_node is async def"
    else
        echo "✗ FAIL: ${node}_node not async def"
    fi
done
echo ""

# Check 3: Verify StateGraph uses START, END
echo "[3] Checking StateGraph imports..."
if grep -q "from langgraph.graph import.*START.*END" /Users/vedantdesai/Projects/Lead-Gen\ Agentic\ Pipeline/src/graph.py; then
    echo "✓ PASS: StateGraph imports START, END"
else
    echo "✗ FAIL: Missing START, END imports"
fi
echo ""

# Check 4: Verify graph.compile() is used (not async compile)
echo "[4] Checking graph compilation..."
if grep -q "compiled_graph = workflow.compile()" /Users/vedantdesai/Projects/Lead-Gen\ Agentic\ Pipeline/src/graph.py; then
    echo "✓ PASS: Using synchronous compile()"
else
    echo "✗ FAIL: Incorrect graph compilation"
fi
echo ""

# Check 5: Verify Annotated reducers for state arrays
echo "[5] Checking Annotated state reducers..."
if grep -q "Annotated\[List\[str\], operator.add\]" /Users/vedantdesai/Projects/Lead-Gen\ Agentic\ Pipeline/src/models/state.py; then
    echo "✓ PASS: Annotated reducers in state"
else
    echo "✗ FAIL: Missing Annotated reducers"
fi
echo ""

# Check 6: Verify Node 3 has no LLM (deterministic only)
echo "[6] Checking Node 3 determinism..."
if grep -q "def fuzzy_match" /Users/vedantdesai/Projects/Lead-Gen\ Agentic\ Pipeline/src/nodes/consensus.py; then
    echo "✓ PASS: Uses fuzzy_match (deterministic)"
else
    echo "✗ FAIL: Missing fuzzy_match"
fi

if grep -q "llm_service\|extract_structured" /Users/vedantdesai/Projects/Lead-Gen\ Agentic\ Pipeline/src/nodes/consensus.py; then
    echo "✗ FAIL: Node 3 still calls LLM"
else
    echo "✓ PASS: Node 3 is pure Python (no LLM)"
fi
echo ""

# Check 7: Verify mock implementations exist
echo "[7] Checking mock implementations..."
for service in tavily crawl4ai hunter; do
    if grep -q "_mock" /Users/vedantdesai/Projects/Lead-Gen\ Agentic\ Pipeline/src/services/${service}_service.py; then
        echo "✓ PASS: ${service}_service has mocks"
    else
        echo "✗ FAIL: ${service}_service missing mocks"
    fi
done
echo ""

echo "=========================================="
echo "ASYNC VERIFICATION COMPLETE"
echo "=========================================="
echo ""
echo "To test with mocks:"
echo "  export USE_MOCKS=true"
echo "  python -m src.main"
echo ""
echo "To test with real APIs:"
echo "  export GOOGLE_API_KEY=your-key"
echo "  export TAVILY_API_KEY=your-key"
echo "  export HUNTER_API_KEY=your-key"
echo "  export USE_MOCKS=false"
echo "  python -m src.main"
