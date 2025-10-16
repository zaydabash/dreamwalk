#!/bin/bash

# DreamWalk Setup Script
# This script sets up the complete DreamWalk system

set -e

echo "Setting up DreamWalk - Neural Dreamscape Generator"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_header "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed"
}

# Check if Python is installed
check_python() {
    print_header "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Python $PYTHON_VERSION is installed"
}

# Create necessary directories
create_directories() {
    print_header "Creating project directories..."
    
    mkdir -p models/checkpoints
    mkdir -p models/exports
    mkdir -p datasets/synthetic
    mkdir -p datasets/real
    mkdir -p unity/DreamWalkVR/Assets/Generated
    mkdir -p unity/builds
    mkdir -p logs
    
    print_status "Project directories created"
}

# Setup environment file
setup_environment() {
    print_header "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            print_status "Environment file created from template"
            print_warning "Please edit .env file with your API keys and configuration"
        else
            print_error "env.example file not found"
            exit 1
        fi
    else
        print_status "Environment file already exists"
    fi
}

# Install Python dependencies
install_python_dependencies() {
    print_header "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements for each service
    for service in services/*/; do
        if [ -f "$service/requirements.txt" ]; then
            print_status "Installing dependencies for $(basename $service)..."
            pip install -r "$service/requirements.txt"
        fi
    done
    
    # Install additional dependencies for scripts
    pip install redis httpx numpy pandas
    
    print_status "Python dependencies installed"
}

# Setup Unity project (if Unity is available)
setup_unity() {
    print_header "Setting up Unity project..."
    
    if command -v unity &> /dev/null || command -v /Applications/Unity/Hub/Editor/*/Unity.app/Contents/MacOS/Unity &> /dev/null; then
        print_status "Unity found - Unity project setup available"
        print_warning "Please open Unity and import the DreamWalkVR project manually"
    else
        print_warning "Unity not found - Unity project setup skipped"
        print_warning "Please install Unity 2023.3+ and import the project manually"
    fi
}

# Build Docker images
build_docker_images() {
    print_header "Building Docker images..."
    
    # Build images for each service
    docker-compose build
    
    print_status "Docker images built successfully"
}

# Start services
start_services() {
    print_header "Starting DreamWalk services..."
    
    # Start all services
    docker-compose up -d
    
    print_status "Services started"
    print_status "Waiting for services to be ready..."
    
    # Wait for services to be ready
    sleep 30
    
    # Check service health
    check_service_health
}

# Check service health
check_service_health() {
    print_header "Checking service health..."
    
    services=(
        "http://localhost:8000"  # Web Dashboard
        "http://localhost:8001"  # Signal Processor
        "http://localhost:8002"  # Neural Decoder
        "http://localhost:8003"  # Real-time Server
        "http://localhost:8005"  # Texture Generator
        "http://localhost:8006"  # Narrative Layer
    )
    
    for service in "${services[@]}"; do
        if curl -s -f "$service/health" > /dev/null; then
            print_status "[OK] $service is healthy"
        else
            print_warning "[ERROR] $service is not responding"
        fi
    done
}

# Setup monitoring
setup_monitoring() {
    print_header "Setting up monitoring..."
    
    # Check if Grafana is accessible
    if curl -s -f "http://localhost:3000" > /dev/null; then
        print_status "[OK] Grafana is accessible at http://localhost:3000"
        print_warning "Default login: admin/admin"
    else
        print_warning "[ERROR] Grafana is not accessible"
    fi
    
    # Check if Prometheus is accessible
    if curl -s -f "http://localhost:9090" > /dev/null; then
        print_status "[OK] Prometheus is accessible at http://localhost:9090"
    else
        print_warning "[ERROR] Prometheus is not accessible"
    fi
}

# Run demo
run_demo() {
    print_header "Running DreamWalk demo..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start mock EEG stream in background
    print_status "Starting mock EEG data stream..."
    python scripts/mock_eeg_stream.py --session-id demo_session --duration 60 &
    EEG_PID=$!
    
    print_status "Demo is running"
    print_status "Open http://localhost:8000 in your browser to view the dashboard"
    print_status "Unity VR project is available in unity/DreamWalkVR/"
    
    # Wait for demo to complete
    wait $EEG_PID
    
    print_status "Demo completed"
}

# Main setup function
main() {
    echo "Starting DreamWalk setup..."
    echo ""
    
    # Run setup steps
    check_docker
    check_python
    create_directories
    setup_environment
    install_python_dependencies
    setup_unity
    build_docker_images
    start_services
    setup_monitoring
    
    echo ""
    echo "DreamWalk setup completed successfully"
    echo ""
    echo "Next steps:"
    echo "1. Open http://localhost:8000 for the web dashboard"
    echo "2. Open http://localhost:3000 for Grafana monitoring (admin/admin)"
    echo "3. Run: ./run_demo.sh to start a demo session"
    echo "4. Import unity/DreamWalkVR/ into Unity for VR experience"
    echo ""
    echo "For more information, see README.md"
}

# Handle script arguments
case "${1:-}" in
    "demo")
        run_demo
        ;;
    "health")
        check_service_health
        ;;
    "stop")
        print_header "Stopping DreamWalk services..."
        docker-compose down
        print_status "Services stopped"
        ;;
    "restart")
        print_header "Restarting DreamWalk services..."
        docker-compose restart
        print_status "Services restarted"
        ;;
    "logs")
        print_header "Showing service logs..."
        docker-compose logs -f
        ;;
    "clean")
        print_header "Cleaning up DreamWalk..."
        docker-compose down -v
        docker system prune -f
        print_status "Cleanup completed"
        ;;
    *)
        main
        ;;
esac
