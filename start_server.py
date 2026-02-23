#!/usr/bin/env python3
"""
Simple HTTP server to view ProjectionAI results
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler

# Serve from the ProjectionAI directory
os.chdir('/home/futurepr0n/Development/ProjectionAI')

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass

with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
    print(f"🌐 ProjectionAI Results Dashboard")
    print(f"   Running at: http://localhost:{PORT}")
    print(f"   Report available at: http://localhost:{PORT}/results_report.html")
    print(f"\n   Press Ctrl+C to stop the server\n")

    # Open browser automatically
    webbrowser.open(f'http://localhost:{PORT}/results_report.html')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        sys.exit(0)
