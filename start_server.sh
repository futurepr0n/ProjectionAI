#!/bin/bash
# Start the ProjectionAI dashboard server

cd /home/futurepr0n/Development/ProjectionAI

# Kill any existing instances
pkill -9 -f "dashboards/app.py" 2>/dev/null
sleep 2

# Start server in background
nohup python3 dashboards/app.py > /tmp/projectionai.log 2>&1 &
PID=$!
echo "Started server with PID: $PID"

# Wait for startup
sleep 5

# Test endpoint
echo "Testing endpoint..."
curl -s "http://localhost:5002/api/predictions/2025-08-29/with_results" | head -200

echo ""
echo "Server should be running at http://localhost:5002"
