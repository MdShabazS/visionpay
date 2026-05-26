#!/bin/bash
# Double-click this file to start VisionPay real-time detector
cd "$(dirname "$0")"
echo "VisionPay - Real-Time Mode"
echo "==========================="
echo "Loading model... (first launch takes ~10s)"
echo ""
venv/bin/python realtime.py
