import io
import base64
from pathlib import Path
from typing import Optional, Literal
import tempfile
import os
import torch

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torchaudio as ta
import torch
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import gridfs
from pymongo import MongoClient

from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import dotenv

dotenv.load_dotenv()
# ======================= Configuration =======================
MONGODB_URI = dotenv.get_key(".env", "MONGODB_URI")  # Update with your MongoDB URI
DATABASE_NAME = dotenv.get_key(".env", "DATABASE_NAME")
REFERENCE_AUDIO_COLLECTION = dotenv.get_key(".env", "REFERENCE_AUDIO_COLLECTION")
CONDITIONALS_COLLECTION = dotenv.get_key(".env", "CONDITIONALS_COLLECTION")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================= Pydantic Models =======================
class PrepareConditionalsRequest(BaseModel):
    """Request to prepare and cache voice conditionals"""
    name: str = Field(..., description="Unique identifier for the voice")
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0, description="Voice characteristic exaggeration")

class PrepareConditionalsResponse(BaseModel):
    """Response after preparing conditionals"""
    name: str
    message: str
    conditionals_id: str

class GenerateAudioRequest(BaseModel):
    """Request to generate audio"""
    name: str = Field(..., description="Name to lookup reference audio")
    text: str = Field(..., description="Text to convert to speech")
    language: Literal["en", "hi"] = Field(default="en", description="Language code: 'en' for English, 'hi' for Hindi")
    
    # Optional generation parameters
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    min_p: float = Field(default=0.05, ge=0.0, le=1.0)

class UploadReferenceAudioRequest(BaseModel):
    """Request to upload reference audio"""
    name: str = Field(..., description="Unique identifier for the voice")
    audio_base64: str = Field(..., description="Base64 encoded audio file (WAV format)")

# ======================= FastAPI App =======================
app = FastAPI(
    title="Voice Cloning API",
    description="Efficient voice cloning service with MongoDB caching",
    version="1.0.0"
)

# ======================= Global Variables =======================
mongodb_client: Optional[AsyncIOMotorClient] = None
db = None
english_model: Optional[ChatterboxTTS] = None
multilingual_model: Optional[ChatterboxMultilingualTTS] = None

# ======================= Database Functions =======================
async def get_reference_audio(name: str) -> Optional[bytes]:
    """Retrieve reference audio from MongoDB by name"""
    audio_doc = await db[REFERENCE_AUDIO_COLLECTION].find_one({"name": name})
    if audio_doc:
        return audio_doc["audio_data"]
    return None

async def save_reference_audio(name: str, audio_data: bytes) -> str:
    """Save reference audio to MongoDB"""
    existing = await db[REFERENCE_AUDIO_COLLECTION].find_one({"name": name})
    
    doc = {
        "name": name,
        "audio_data": audio_data,
    }
    
    if existing:
        await db[REFERENCE_AUDIO_COLLECTION].update_one(
            {"name": name},
            {"$set": doc}
        )
        return str(existing["_id"])
    else:
        result = await db[REFERENCE_AUDIO_COLLECTION].insert_one(doc)
        return str(result.inserted_id)

async def get_conditionals(name: str):
    """Retrieve cached conditionals from MongoDB"""
    print(f"🔍 Looking for conditionals with name: '{name}'")
    cond_doc = await db[CONDITIONALS_COLLECTION].find_one({"name": name})
    
    if cond_doc:
        print(f"✅ Found conditionals document: {cond_doc.get('_id')}")
        try:
            # Deserialize conditionals from binary data
            buffer = io.BytesIO(cond_doc["conditionals_data"])
            loaded_data = torch.load(buffer, map_location=DEVICE)
            
            # Reconstruct Conditionals object
            from chatterbox.tts import Conditionals
            conditionals = Conditionals(t3=loaded_data["t3"], gen=loaded_data["gen"])
            
            print(f"✅ Successfully deserialized conditionals for: {name}")
            print(f"   Type: {type(conditionals)}")
            return conditionals
        except Exception as e:
            print(f"❌ Failed to deserialize conditionals: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"❌ No conditionals document found for name: '{name}'")
        return None

async def save_conditionals(name: str, conditionals, exaggeration: float) -> str:
    """Save conditionals to MongoDB"""
    # Serialize conditionals to binary format
    buffer = io.BytesIO()
    torch.save({"t3": conditionals.t3, "gen": conditionals.gen}, buffer)
    buffer.seek(0)
    conditionals_bytes = buffer.read()
    
    existing = await db[CONDITIONALS_COLLECTION].find_one({"name": name})
    
    doc = {
        "name": name,
        "conditionals_data": conditionals_bytes,
        "exaggeration": exaggeration,
    }
    
    if existing:
        await db[CONDITIONALS_COLLECTION].update_one(
            {"name": name},
            {"$set": doc}
        )
        return str(existing["_id"])
    else:
        result = await db[CONDITIONALS_COLLECTION].insert_one(doc)
        return str(result.inserted_id)

# ======================= Startup/Shutdown Events =======================
@app.on_event("startup")
async def startup_event():
    """Initialize models and database connection on startup"""
    global mongodb_client, db, english_model, multilingual_model
    
    print("🚀 Starting Voice Cloning API...")
    
    # Initialize MongoDB
    print("📦 Connecting to MongoDB...")
    mongodb_client = AsyncIOMotorClient(MONGODB_URI)
    db = mongodb_client[DATABASE_NAME]
    print("✅ MongoDB connected")
    
    # Initialize models
    print(f"🤖 Loading models on device: {DEVICE}")
    print("   Loading English model...")
    english_model = ChatterboxTTS.from_pretrained(device=DEVICE)
    print("   ✅ English model loaded")
    
    print("   Loading Multilingual model...")
    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    print("   ✅ Multilingual model loaded")
    
    print("🎉 API Ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        print("👋 MongoDB connection closed")

# ======================= API Endpoints =======================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Voice Cloning API",
        "device": DEVICE,
        "models_loaded": {
            "english": english_model is not None,
            "multilingual": multilingual_model is not None
        }
    }

@app.post("/upload-reference-audio", status_code=status.HTTP_201_CREATED)
async def upload_reference_audio(request: UploadReferenceAudioRequest):
    """
    Upload reference audio to database.
    This should be called first to store the reference voice.
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Save to database
        audio_id = await save_reference_audio(request.name, audio_bytes)
        
        return {
            "message": "Reference audio uploaded successfully",
            "name": request.name,
            "audio_id": audio_id
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload reference audio: {str(e)}"
        )

@app.post("/prepare-conditionals", response_model=PrepareConditionalsResponse)
async def prepare_conditionals(request: PrepareConditionalsRequest):
    """
    Prepare and cache voice conditionals for a reference audio.
    Call this once after uploading reference audio to pre-compute conditionals.
    This improves generation speed for subsequent requests.
    """
    temp_audio_path = None
    try:
        # Check if reference audio exists
        audio_data = await get_reference_audio(request.name)
        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference audio not found for name: {request.name}"
            )
        
        # Use Windows-compatible temp directory
        temp_dir = Path(tempfile.gettempdir())
        temp_audio_path = temp_dir / f"{request.name}_ref.wav"
        
        print(f"🔧 Preparing conditionals for: {request.name}")
        print(f"   Temp directory: {temp_dir}")
        print(f"   Audio path: {temp_audio_path}")
        print(f"   Audio path exists after write: {temp_audio_path.exists()}")
        
        # Write audio data
        temp_audio_path.write_bytes(audio_data)
        print(f"   Audio file size: {temp_audio_path.stat().st_size} bytes")
        
        # Verify it's a valid WAV file
        try:
            waveform, sample_rate = ta.load(str(temp_audio_path))
            print(f"   Audio loaded successfully:")
            print(f"     - Sample rate: {sample_rate}")
            print(f"     - Shape: {waveform.shape}")
            print(f"     - Duration: {waveform.shape[1] / sample_rate:.2f}s")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid audio file format: {str(e)}" 
            )
        
        # Prepare conditionals using English model
        # Note: prepare_conditionals doesn't return a value, it sets model.conds internally
        print(f"   Calling prepare_conditionals with exaggeration={request.exaggeration}")
        english_model.prepare_conditionals(
            wav_fpath=str(temp_audio_path),
            exaggeration=request.exaggeration
        )
        
        # Get the conditionals from the model
        conditionals = english_model.conds
        
        print(f"   Conditionals type: {type(conditionals)}")
        print(f"   Conditionals is None: {conditionals is None}")
        
        # Verify conditionals is not None
        if conditionals is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate conditionals - returned None. Check audio file format and quality."
            )
        
        # Save to database
        cond_id = await save_conditionals(request.name, conditionals, request.exaggeration)
        
        # Cleanup temp file
        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink()
        
        print(f"✅ Conditionals prepared and cached for: {request.name}")
        
        return PrepareConditionalsResponse(
            name=request.name,
            message="Conditionals prepared and cached successfully",
            conditionals_id=cond_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prepare conditionals: {str(e)}"
        )
    finally:
        # Ensure cleanup
        if temp_audio_path and temp_audio_path.exists():
            try:
                temp_audio_path.unlink()
            except:
                pass

@app.post("/generate-audio")
async def generate_audio(request: GenerateAudioRequest):
    """
    Generate audio from text using cached conditionals or reference audio.
    """
    temp_audio_path = None
    try:
        # Clean the text
        clean_text = punc_norm(request.text)
        
        # Try to get cached conditionals first (fast path)
        conditionals = await get_conditionals(request.name)
        
        # Better check: verify conditionals is not None
        has_conditionals = conditionals is not None
        print(f"🔍 Conditionals check: {has_conditionals}, Type: {type(conditionals) if conditionals else 'None'}")
        
        if not has_conditionals:
            # Fallback: get reference audio
            print(f"⚠️  No cached conditionals for {request.name}, using reference audio")
            audio_data = await get_reference_audio(request.name)
            if not audio_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Neither conditionals nor reference audio found for: {request.name}. "
                           "Please upload reference audio first using /upload-reference-audio"
                )
            
            # Use Windows-compatible temp directory
            temp_dir = Path(tempfile.gettempdir())
            temp_audio_path = temp_dir / f"{request.name}_ref.wav"
            temp_audio_path.write_bytes(audio_data)
            audio_prompt_path = str(temp_audio_path)
        else:
            print(f"✅ Using cached conditionals for: {request.name}")
            audio_prompt_path = None
        
        # Select appropriate model based on language
        if request.language == "en":
            model = english_model
            print(f"🎙️  Generating English audio for: {request.name}")
            
            if has_conditionals:
                # Use cached conditionals
                model.conds = conditionals
                wav = model.generate(
                    text=clean_text,
                    temperature=request.temperature,
                    repetition_penalty=request.repetition_penalty,
                    cfg_weight=request.cfg_weight,
                    exaggeration=request.exaggeration,
                    top_p=request.top_p,
                    min_p=request.min_p
                )
            else:
                # Use reference audio
                wav = model.generate(
                    text=clean_text,
                    audio_prompt_path=audio_prompt_path,
                    temperature=request.temperature,
                    repetition_penalty=request.repetition_penalty,
                    cfg_weight=request.cfg_weight,
                    exaggeration=request.exaggeration,
                    top_p=request.top_p,
                    min_p=request.min_p
                )
        
        elif request.language == "hi":
            model = multilingual_model
            print(f"🎙️  Generating Hindi audio for: {request.name}")
            
            # Note: Multilingual model uses audio_prompt_path directly
            # Check if we need to create temp file from conditionals
            if has_conditionals and audio_prompt_path is None:
                # For multilingual, we still need the audio file
                audio_data = await get_reference_audio(request.name)
                if audio_data:
                    temp_dir = Path(tempfile.gettempdir())
                    temp_audio_path = temp_dir / f"{request.name}_ref.wav"
                    temp_audio_path.write_bytes(audio_data)
                    audio_prompt_path = str(temp_audio_path)
            
            wav = model.generate(
                text=clean_text,
                audio_prompt_path=audio_prompt_path,
                language_id=request.language,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                cfg_weight=request.cfg_weight,
                exaggeration=request.exaggeration,
                top_p=request.top_p,
                min_p=request.min_p
            )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported language: {request.language}"
            )
        
        # Convert tensor to WAV bytes
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        print(f"✅ Audio generated successfully for: {request.name}")
        
        # Return as streaming response
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{request.name}_{request.language}.wav"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate audio: {str(e)}"
        )
    finally:
        # Cleanup temp file if created
        if temp_audio_path and temp_audio_path.exists():
            try:
                temp_audio_path.unlink()
            except:
                pass

@app.delete("/delete-voice/{name}")
async def delete_voice(name: str):
    """Delete reference audio and conditionals for a given name"""
    try:
        audio_result = await db[REFERENCE_AUDIO_COLLECTION].delete_one({"name": name})
        cond_result = await db[CONDITIONALS_COLLECTION].delete_one({"name": name})
        
        if audio_result.deleted_count == 0 and cond_result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data found for name: {name}"
            )
        
        return {
            "message": f"Deleted voice data for: {name}",
            "audio_deleted": audio_result.deleted_count > 0,
            "conditionals_deleted": cond_result.deleted_count > 0
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete voice: {str(e)}"
        )

@app.get("/list-voices")
async def list_voices():
    """List all available voices in the database"""
    try:
        # Get all unique voice names
        audio_names = await db[REFERENCE_AUDIO_COLLECTION].distinct("name")
        cond_names = await db[CONDITIONALS_COLLECTION].distinct("name")
        
        # Create a comprehensive list
        voices = []
        all_names = set(audio_names + cond_names)
        
        for name in all_names:
            has_audio = name in audio_names
            has_conditionals = name in cond_names
            
            voices.append({
                "name": name,
                "has_reference_audio": has_audio,
                "has_conditionals": has_conditionals,
                "ready": has_audio and has_conditionals
            })
        
        return {
            "total_voices": len(voices),
            "voices": voices
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list voices: {str(e)}"
        )

# ======================= Main =======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
