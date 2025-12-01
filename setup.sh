#!/bin/bash

# Movement Chain ML - Development Environment Setup Script
# This script automates the setup of Git hooks and development dependencies

set -e  # Exit on error

echo "🚀 Setting up Movement Chain ML development environment..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    echo "Please install Node.js 20+ from https://nodejs.org/"
    exit 1
fi

echo -e "${GREEN}✅ Node.js found: $(node --version)${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python is not installed${NC}"
    echo "Please install Python 3.11+ from https://python.org/"
    exit 1
fi

echo -e "${GREEN}✅ Python found: $(python3 --version)${NC}"
echo ""

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Node.js dependencies installed${NC}"
else
    echo -e "${RED}❌ Failed to install Node.js dependencies${NC}"
    exit 1
fi

echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

echo ""
echo "📦 Installing Python development dependencies..."
echo -e "${YELLOW}Please activate the virtual environment and run:${NC}"
echo ""
echo "  source venv/bin/activate  # On Unix/macOS"
echo "  # OR"
echo "  venv\\Scripts\\activate     # On Windows"
echo ""
echo "  pip install -r requirements-dev.txt"
echo ""

# Make hooks executable
echo "🔧 Making Git hooks executable..."
chmod +x .husky/commit-msg
chmod +x .husky/pre-commit
chmod +x .husky/pre-push
echo -e "${GREEN}✅ Git hooks are executable${NC}"

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Install Python dependencies:"
echo "   pip install -r requirements-dev.txt"
echo ""
echo "3. Verify installation:"
echo "   black --version"
echo "   ruff --version"
echo "   pytest --version"
echo ""
echo "4. Read the documentation:"
echo "   cat HOOKS_SETUP.md"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
