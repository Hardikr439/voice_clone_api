# Voice Cloning API

FastAPI service for voice cloning with MongoDB caching. Supports English and Hindi.

## Requirements

- Python 3.9

## Quick Start

```bash
# Install dependencies
pip install -r requirements_api.txt

# Start MongoDB
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Start API
python api.py

# Test
python test_client.py
```

## Workflow

1. **Upload reference audio** → 2. **Prepare conditionals** (one-time) → 3. **Generate audio** (10x faster!)

## Main Endpoints

### Upload Reference Audio

```http
POST /upload-reference-audio
```

```json
{ "name": "john_doe", "audio_base64": "UklGRiQAAAB..." }
```

### Prepare Conditionals (Recommended - Call Once)

```http
POST /prepare-conditionals
```

```json
{ "name": "john_doe", "exaggeration": 0.5 }
```

### Generate Audio

```http
POST /generate-audio
```

```json
{
  "name": "john_doe",
  "text": "Your text here",
  "language": "en"
}
```

**Optional parameters:** `temperature`, `repetition_penalty`, `cfg_weight`, `exaggeration`, `top_p`, `min_p`

**Response:** WAV audio file

### Other Endpoints

- `GET /list-voices` - List all voices
- `DELETE /delete-voice/{name}` - Delete voice data

## Python Example

```python
import requests, base64

API_URL = "http://localhost:8000"

# Upload
with open("voice.wav", "rb") as f:
    requests.post(f"{API_URL}/upload-reference-audio",
                  json={"name": "my_voice", "audio_base64": base64.b64encode(f.read()).decode()})

# Prepare (one-time)
requests.post(f"{API_URL}/prepare-conditionals", json={"name": "my_voice"})

# Generate
response = requests.post(f"{API_URL}/generate-audio",
                         json={"name": "my_voice", "text": "Hello!", "language": "en"})
open("output.wav", "wb").write(response.content)
```

## Configuration

Edit `api.py`:

```python
MONGODB_URI = "mongodb://localhost:27017"
DEVICE = "cuda"  # or "cpu"
```
