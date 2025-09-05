#!/bin/bash
# Test runner script for local development

set -e  # Exit on any error

echo "🧪 Running Local GenAI Test Suite"
echo "================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider running: conda activate tf_gemma (or your env name)"
fi

# Install test dependencies if needed
echo "📦 Installing test dependencies..."
pip install -r requirements-test.txt --quiet

# Run code quality checks
echo ""
echo "🔍 Running code quality checks..."

echo "  → Checking code formatting with black..."
if black --check scripts tests --quiet; then
    echo "    ✅ Code formatting passed"
else
    echo "    ❌ Code formatting failed. Run: black scripts tests"
    exit 1
fi

echo "  → Checking import sorting with isort..."
if isort --check-only scripts tests --quiet; then
    echo "    ✅ Import sorting passed"
else
    echo "    ❌ Import sorting failed. Run: isort scripts tests"
    exit 1
fi

echo "  → Running flake8 linting..."
if flake8 scripts tests --quiet; then
    echo "    ✅ Linting passed"
else
    echo "    ❌ Linting failed"
    exit 1
fi

# Run tests with different categories
echo ""
echo "🧪 Running unit tests..."

# Fast unit tests first
echo "  → Running fast unit tests..."
pytest tests/ -v -m "not slow and not gpu and not network" \
    --tb=short \
    --durations=10 \
    --cov=scripts \
    --cov-report=term-missing

echo ""
echo "🎯 Test Results Summary:"
echo "  → Check htmlcov/index.html for detailed coverage report"
echo "  → Run specific tests: pytest tests/test_simple_rag.py -v"
echo "  → Run with specific markers: pytest -m unit -v"
echo ""

# Optional: Run slow tests if requested
if [[ "$1" == "--include-slow" ]]; then
    echo "⏳ Running slow tests..."
    pytest tests/ -v -m "slow" --tb=short
fi

# Optional: Run GPU tests if requested and available
if [[ "$1" == "--include-gpu" ]]; then
    echo "🖥️  Running GPU tests..."
    pytest tests/ -v -m "gpu" --tb=short
fi

echo "✅ All tests completed successfully!"