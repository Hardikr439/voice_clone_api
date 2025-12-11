"""
Test client for Voice Cloning API
Demonstrates how to use all endpoints
"""

import requests
import base64
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_health():
    """Check if API is running"""
    response = requests.get(f"{API_BASE_URL}/")
    print("🏥 Health Check:", response.json())
    return response.status_code == 200

def upload_reference_audio(name: str, audio_file_path: str):
    """Upload reference audio file"""
    print(f"\n📤 Uploading reference audio for: {name}")
    
    # Read and encode audio file
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        return None
    
    audio_bytes = audio_path.read_bytes()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Send request
    response = requests.post(
        f"{API_BASE_URL}/upload-reference-audio",
        json={
            "name": name,
            "audio_base64": audio_base64
        }
    )
    
    if response.status_code == 201:
        print("✅ Upload successful:", response.json())
        return response.json()
    else:
        print("❌ Upload failed:", response.text)
        return None

def prepare_conditionals(name: str, exaggeration: float = 0.5):
    """Prepare and cache conditionals for a voice"""
    print(f"\n🔧 Preparing conditionals for: {name}")
    
    response = requests.post(
        f"{API_BASE_URL}/prepare-conditionals",
        json={
            "name": name,
            "exaggeration": exaggeration
        }
    )
    
    if response.status_code == 200:
        print("✅ Conditionals prepared:", response.json())
        return response.json()
    else:
        print("❌ Failed to prepare conditionals:", response.text)
        return None

def generate_audio(
    name: str,
    text: str,
    language: str = "en",
    output_file: str = "output.wav",
    **kwargs
):
    """Generate audio from text"""
    print(f"\n🎙️  Generating audio for: {name}")
    print(f"   Text: {text[:50]}...")
    print(f"   Language: {language}")
    
    payload = {
        "name": name,
        "text": text,
        "language": language,
        **kwargs
    }
    
    response = requests.post(
        f"{API_BASE_URL}/generate-audio",
        json=payload
    )
    
    if response.status_code == 200:
        # Save audio file
        output_path = Path(output_file)
        output_path.write_bytes(response.content)
        print(f"✅ Audio saved to: {output_path.absolute()}")
        return str(output_path.absolute())
    else:
        print("❌ Failed to generate audio:", response.text)
        return None

def list_voices():
    """List all available voices"""
    print("\n📋 Listing all voices:")
    
    response = requests.get(f"{API_BASE_URL}/list-voices")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n   Total voices: {data['total_voices']}")
        for voice in data['voices']:
            status = "✅ Ready" if voice['ready'] else "⚠️  Incomplete"
            print(f"   {status} - {voice['name']}")
            print(f"      Has audio: {voice['has_reference_audio']}, Has conditionals: {voice['has_conditionals']}")
        return data
    else:
        print("❌ Failed to list voices:", response.text)
        return None

def delete_voice(name: str):
    """Delete a voice from database"""
    print(f"\n🗑️  Deleting voice: {name}")
    
    response = requests.delete(f"{API_BASE_URL}/delete-voice/{name}")
    
    if response.status_code == 200:
        print("✅ Deleted:", response.json())
        return response.json()
    else:
        print("❌ Failed to delete:", response.text)
        return None

# ======================= Example Usage =======================

def example_workflow():
    """Complete workflow example"""
    print("=" * 60)
    print("🚀 Voice Cloning API - Example Workflow")
    print("=" * 60)
    
    # 1. Check health
    if not check_health():
        print("❌ API is not running. Please start it first.")
        return
    
    # 2. Upload reference audio
    VOICE_NAME = "striver"
    REFERENCE_AUDIO = "WhatsApp Ptt 2025-12-11 at 01.32.52.wav"  # Update with your file
    
    print("\n" + "=" * 60)
    print("Step 1: Upload Reference Audio")
    print("=" * 60)
    upload_reference_audio(VOICE_NAME, REFERENCE_AUDIO)
    
    # 3. Prepare conditionals (one-time setup)
    print("\n" + "=" * 60)
    print("Step 2: Prepare Conditionals (One-time Setup)")
    print("=" * 60)
    prepare_conditionals(VOICE_NAME, exaggeration=0.7)
    
    # 4. List voices
    print("\n" + "=" * 60)
    print("Step 3: List Available Voices")
    print("=" * 60)
    list_voices()
    
    # 5. Generate English audio
    print("\n" + "=" * 60)
    print("Step 4: Generate English Audio")
    print("=" * 60)
    english_text = "Hello! This is a test of the voice cloning system. It's working perfectly!"
    generate_audio(
        name=VOICE_NAME,
        text=english_text,
        language="en",
        output_file="output_english.wav",
        temperature=0.75,
        repetition_penalty=1.3
    )
    
    # 6. Generate Hindi audio
    print("\n" + "=" * 60)
    print("Step 5: Generate Hindi Audio")
    print("=" * 60)
    hindi_text = "नमस्ते, आज का दिन बहुत अच्छा है।"
    generate_audio(
        name=VOICE_NAME,
        text=hindi_text,
        language="hi",
        output_file="output_hindi.wav"
    )
    
    # 7. Generate another audio (uses cached conditionals - faster!)
    print("\n" + "=" * 60)
    print("Step 6: Generate Another Audio (Using Cached Conditionals)")
    print("=" * 60)
    tech_text = "DFS, or Depth-First Search, is a graph-traversal algorithm that explores as far as possible along each path before backtracking."
    generate_audio(
        name=VOICE_NAME,
        text=tech_text,
        language="en",
        output_file="output_tech.wav"
    )
    
    print("\n" + "=" * 60)
    print("✅ Workflow Complete!")
    print("=" * 60)

if __name__ == "__main__":
    example_workflow()
