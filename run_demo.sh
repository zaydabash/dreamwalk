#!/bin/bash

# DreamWalk Demo Runner
# This script runs a complete demo of the DreamWalk system

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[DEMO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_header "Starting DreamWalk Demo"
echo "================================"

# Check if services are running
print_status "Checking if services are running..."

if ! curl -s -f "http://localhost:8003/health" > /dev/null; then
    print_warning "Services are not running. Starting them..."
    docker-compose up -d
    print_status "Waiting for services to start..."
    sleep 30
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    print_status "Virtual environment activated"
fi

# Start mock EEG stream
print_header "Starting mock EEG data stream..."
print_status "Generating realistic synthetic EEG data for 2 minutes"
print_status "Watch the dashboard at http://localhost:8000"

python scripts/mock_eeg_stream.py \
    --session-id "demo_session_$(date +%s)" \
    --duration 120 \
    --update-rate 10

print_status "Demo completed"
print_status "You can view the dashboard at http://localhost:8000"
print_status "Check the Unity VR project in unity/DreamWalkVR/ for immersive experience"
