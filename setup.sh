#!/bin/bash
# ProjectionAI - Setup Script

set -e

echo "🚀 Setting up ProjectionAI..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install baseball_scraper for Statcast data
echo "⚾ Installing baseball_scraper..."
pip install baseball_scraper

# Check PostgreSQL
echo "🗄️  Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "✅ PostgreSQL found"
    echo "💡 Create database: createdb projectionai"
else
    echo "⚠️  PostgreSQL not found. Install: sudo apt-get install postgresql postgresql-contrib"
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p data models dashboards notebooks docs logs

# Set up logging
mkdir -p logs
touch logs/projectionai.log

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Create database: createdb projectionai"
echo "  3. Run database setup: python data/database.py"
echo "  4. Start working on models: python models/train.py"
echo ""
