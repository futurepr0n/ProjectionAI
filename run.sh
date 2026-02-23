#!/bin/bash
cd /home/futurepr0n/Development/ProjectionAI
pkill -9 -f "dashboards/app.py" 2>/dev/null
sleep 2
nohup python3 dashboards/app.py > /tmp/pai.log 2>&1 &
sleep 5
echo "Server status:"
curl -s http://localhost:5002/api/model/stats
echo ""
echo "Dashboard: http://localhost:5002"
echo "Network: http://192.168.1.158:5002"
