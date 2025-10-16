# Contributing to DreamWalk

Thank you for your interest in contributing to DreamWalk. This document provides guidelines and information for contributors.

## What is DreamWalk?

DreamWalk is a system that translates neural signals (EEG/fMRI) into dynamic, procedurally generated dreamscapes that you can explore in VR. It combines neuroscience, AI, procedural generation, and immersive interfaces.

## Getting Started

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Unity 2023.3+ (for VR components)
- Git

### Setup Development Environment

1. Fork and clone the repository
2. Run the setup script:
   ```bash
   ./setup.sh
   ```
3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

## Project Structure

```
dreamwalk/
├── services/                 # Microservices
│   ├── signal-processor/     # EEG/fMRI processing
│   ├── neural-decoder/       # Brain-to-latent mapping
│   ├── realtime-server/      # WebSocket orchestration
│   ├── texture-generator/    # Stable Diffusion service
│   ├── narrative-layer/      # LLM narration
│   └── web-dashboard/        # Monitoring UI
├── unity/                   # Unity VR project
├── scripts/                 # Utility scripts
├── monitoring/              # Prometheus/Grafana configs
└── docs/                   # Documentation
```

## Development Workflow

### 1. Feature Development

- Create a feature branch: `git checkout -b feature/your-feature`
- Make your changes
- Add tests for new functionality
- Ensure all tests pass: `docker-compose -f docker-compose.test.yml up`
- Submit a pull request

### 2. Service Development

Each service is independent and can be developed separately:

```bash
# Start specific service
cd services/signal-processor
python main.py

# Run tests
python -m pytest

# Build Docker image
docker build -t dreamwalk-signal-processor .
```

### 3. Unity Development

- Open Unity 2023.3+
- Import the project from `unity/DreamWalkVR/`
- Install required packages:
  - XR Interaction Toolkit
  - Universal Render Pipeline
  - OpenXR Plugin

## Testing

### Unit Tests

```bash
# Run all tests
docker-compose -f docker-compose.test.yml up

# Run specific service tests
cd services/signal-processor
python -m pytest tests/
```

### Integration Tests

```bash
# Start all services
docker-compose up -d

# Run integration tests
python scripts/test_integration.py

# Check service health
curl http://localhost:8003/health
```

### Performance Tests

```bash
# Run performance benchmarks
python scripts/benchmark_performance.py

# Load testing
python scripts/load_test.py --duration 300 --concurrent-sessions 10
```

## Code Quality

### Python Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking
- **pytest** for testing

```bash
# Format code
black services/
isort services/

# Type checking
mypy services/

# Run linting
flake8 services/
```

### C# Code Style (Unity)

- Follow Unity C# coding conventions
- Use meaningful variable names
- Add XML documentation for public APIs
- Keep methods under 50 lines when possible

## Research Contributions

DreamWalk touches on several research areas:

### Neuroscience
- EEG signal processing
- Brain-computer interfaces
- Neural pattern recognition
- Emotional state estimation

### Machine Learning
- Neural decoding
- Multimodal learning
- Real-time inference
- Transfer learning

### Computer Graphics
- Procedural generation
- Real-time rendering
- VR/AR interfaces
- Generative textures

### Human-Computer Interaction
- Brain-computer interfaces
- Immersive experiences
- Real-time adaptation
- Accessibility

## Documentation

### Code Documentation

- Use docstrings for all public functions
- Include type hints for Python code
- Document complex algorithms
- Provide usage examples

### API Documentation

- Document all REST endpoints
- Include request/response examples
- Document WebSocket message formats
- Provide authentication details

### Research Documentation

- Document experimental results
- Include methodology details
- Provide reproducibility instructions
- Cite relevant research

## Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, Docker version
2. **Steps to reproduce**: Clear, numbered steps
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Logs**: Relevant log output
6. **Screenshots**: If applicable

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug

## Environment
- OS: [e.g., macOS 13.0]
- Python: [e.g., 3.9.7]
- Docker: [e.g., 20.10.8]

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Logs
```
[Include relevant log output]
```

## Screenshots
[If applicable]
```

## Feature Requests

I welcome feature requests. Please:

1. Check existing issues first
2. Describe the feature clearly
3. Explain the use case
4. Consider implementation complexity
5. Discuss with maintainers first for large features

### Feature Request Template

```markdown
## Feature Description
Brief description of the feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this be implemented?

## Alternatives Considered
What other approaches were considered?

## Additional Context
Any other relevant information
```


## Getting Help

- **Documentation**: Check the README.md and docs/ folder
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions

## Contribution Areas

I especially welcome contributions in:

### High Priority
- Real-time EEG processing improvements
- Neural decoder accuracy enhancements
- VR performance optimizations
- Documentation improvements

### Medium Priority
- Additional biome types
- New emotion classification models
- Mobile app development
- Testing infrastructure

### Research Areas
- Novel neural decoding approaches
- Brain-computer interface research
- Procedural generation algorithms
- Immersive experience design


