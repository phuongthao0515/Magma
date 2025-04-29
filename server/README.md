# Magma API Service

This directory contains a FastAPI server for the Microsoft Magma multimodal model. The API service provides endpoints for inference using the Magma model, including processing images and generating text and action responses.

## Quick Start

The fastest way to run the server is:

```bash
./magma-server.sh run
```

This command will install the required dependencies and start the server directly.

## Directory Structure

The server is organized as follows:
- `main.py` - Main FastAPI application for the Magma model
- `magma-server.sh` - Unified script for managing all deployment methods
- `test_api.py` - Script for testing the API
- `/docker/` - Files for Docker-based deployment
  - `Dockerfile` - Container definition
  - `docker-compose.yml` - Docker Compose configuration
- `/native/` - Files for native system service deployment
  - `run_magma_api.sh` - Script to run the API directly
  - `manage_magma_service.sh` - Script to manage the service
  - `magma-api.service` - Systemd service definition

## API Functionality

This API provides:
- Vision and language processing via a REST API
- Action prediction for robotics applications
- Health check endpoint
- Support for both base64-encoded images and file uploads

## Installation & Usage

This server leverages the main Magma package and installs server-specific dependencies through the optional `[server]` dependencies in pyproject.toml. All server-specific dependencies are defined in one place, avoiding duplicate requirements files.

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.10+

### Using the Unified Management Script

We provide a unified script to manage all deployment methods:

```bash
# Quick start - simplest option
./magma-server.sh run            # Install dependencies and run directly

# For Docker deployment
./magma-server.sh docker up      # Start Docker container
./magma-server.sh docker down    # Stop Docker container
./magma-server.sh docker logs    # View Docker logs
./magma-server.sh docker build   # Build Docker image

# For native deployment
./magma-server.sh native setup    # Set up conda environment
./magma-server.sh native install  # Install as systemd service
./magma-server.sh native start    # Start the service
./magma-server.sh native stop     # Stop the service
./magma-server.sh native run      # Run directly without service
```

### Manual Usage

#### Option 1: Using Docker

1. Navigate to the docker directory:
   ```bash
   cd docker
   ```

2. Build and start the container:
   ```bash
   docker compose up -d
   ```

3. Check the logs:
   ```bash
   docker compose logs -f
   ```

#### Option 2: Running Directly

1. Install the package with server dependencies:
   ```bash
   # From the repository root
   pip install -e ".[server]"
   ```

2. Run the server directly:
   ```bash
   cd server
   python main.py
   ```

3. Test the API (in a separate terminal):
   ```bash
   cd server
   # Basic test - just check if the server is running
   ./test_api.py --url http://localhost:8080
   
   # Full test with an image
   ./test_api.py --url http://localhost:8080 --image /path/to/image.jpg
   ```

#### Option 3: Installing as a Service

1. Navigate to the native directory:
   ```bash
   cd native
   ```

2. Setup the conda environment:
   ```bash
   ./manage_magma_service.sh setup-conda
   ```

3. Install and start the service:
   ```bash
   sudo ./manage_magma_service.sh install
   sudo ./manage_magma_service.sh start
   ```

## API Endpoints

The API will be available on port 8080 by default.

### Health Check
```
GET /health
```

### Predict from Base64 Image
```
POST /predict
```

Request body:
```json
{
  "image": "base64_encoded_image_data",
  "prompt": "What can you see in this image and what action should I take?"
}
```

### Predict from File Upload
```
POST /predict_from_file
```
Use multipart/form-data with:
- `file`: Image file
- `prompt`: Text prompt

### Response Format

```json
{
  "text_response": "Text description from the model",
  "normalized_actions": [-0.25, 0.42, 0.13, 0.0, 0.0, 0.0, 1.0],
  "delta_values": [-0.025, 0.042, 0.013, 0.0, 0.0, 0.0, 1.0]
}
```