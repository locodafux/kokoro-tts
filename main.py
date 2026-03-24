#!/usr/bin/env python3
"""
Kokoro TTS FastAPI Application - Chapter Mode with Automatic MP3 Output
"""

import os
import sys
import warnings
import numpy as np
import soundfile as sf
from pathlib import Path
import subprocess
import tempfile
import torch
import re
import time
import uuid
import shutil
from typing import List, Optional, Union, Dict, Any, Tuple, AsyncGenerator
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn
import logging
from io import BytesIO
import asyncio
import json

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import kokoro with error handling
try:
    from kokoro import KPipeline
except ImportError as e:
    logger.error(f"Error importing kokoro: {e}")
    logger.error("Please install kokoro with: pip install kokoro")
    sys.exit(1)

# Configuration
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Store active files to prevent premature cleanup
active_files = set()

# Pydantic models
class ChapterRequest(BaseModel):
    """Request model for chapter text"""
    title: Optional[str] = Field(None, description="Chapter title")
    content: str = Field(..., description="Chapter content/text")
    voice: str = Field("am_adam", description="Voice to use for synthesis")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed (0.5 to 2.0)")
    add_silence_between_sentences: bool = Field(True, description="Add silence between sentences")
    silence_duration_ms: int = Field(100, ge=0, le=1000, description="Silence duration in milliseconds")
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Chapter content cannot be empty')
        return v.strip()

class StreamingChapterRequest(BaseModel):
    """Request model for streaming chapter text with chunking options"""
    title: Optional[str] = Field(None, description="Chapter title")
    content: str = Field(..., description="Chapter content/text")
    voice: str = Field("am_adam", description="Voice to use for synthesis")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed (0.5 to 2.0)")
    chunk_size: str = Field("sentence", description="Chunking strategy: 'sentence', 'paragraph', or 'character'")
    add_silence_between_chunks: bool = Field(True, description="Add silence between chunks")
    silence_duration_ms: int = Field(100, ge=0, le=1000, description="Silence duration in milliseconds")
    stream_metadata: bool = Field(True, description="Include metadata in the stream")
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Chapter content cannot be empty')
        return v.strip()
    
    @field_validator('chunk_size')
    @classmethod
    def valid_chunk_size(cls, v):
        if v not in ['sentence', 'paragraph', 'character']:
            raise ValueError('chunk_size must be one of: sentence, paragraph, character')
        return v

class BookRequest(BaseModel):
    """Request model for multiple chapters"""
    title: str = Field(..., description="Book title")
    chapters: List[ChapterRequest] = Field(..., description="List of chapters")
    voice: str = Field("am_adam", description="Default voice for all chapters")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Default speech speed")
    add_chapter_markers: bool = Field(True, description="Add audio markers between chapters")
    
    @field_validator('chapters')
    @classmethod
    def chapters_not_empty(cls, v):
        if not v:
            raise ValueError('At least one chapter is required')
        return v

class TTSResponse(BaseModel):
    success: bool
    file_id: str
    filename: str
    duration: float
    sentences: int
    chapters: Optional[int] = None
    message: str

class VoicesResponse(BaseModel):
    voices: List[str]
    default: str = "am_adam"
    recommended_for_narration: List[str] = ["am_adam", "am_michael", "am_leo"]

class StreamingChunk(BaseModel):
    """Model for streaming chunks"""
    type: str  # 'audio' or 'metadata'
    data: Optional[str] = None  # base64 encoded audio for 'audio' type
    index: Optional[int] = None
    text: Optional[str] = None
    duration: Optional[float] = None
    timestamp: Optional[float] = None
    is_final: Optional[bool] = False

class KokoroTTS:
    """Kokoro TTS wrapper class with advanced chapter support"""
    
    def __init__(self, device='cpu', voice='am_adam', repo_id='hexgrad/Kokoro-82M'):
        self.device = device
        self.voice = voice
        self.repo_id = repo_id
        
        # Check if CUDA is available
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        logger.info(f"Initializing Kokoro TTS with device: {self.device}")
        logger.info(f"Using voice: {voice}")
        
        try:
            self.pipeline = KPipeline(lang_code='a', device=self.device, repo_id=repo_id)
            self.sample_rate = 24000
            logger.info(f"Using sample rate: {self.sample_rate} Hz")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling of paragraphs"""
        # Handle paragraphs first
        paragraphs = text.split('\n\n')
        
        sentences = []
        for paragraph in paragraphs:
            # Split into sentences
            paragraph_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
            sentences.extend([s.strip() for s in paragraph_sentences if s.strip()])
        
        # Merge very short sentences (likely part of previous sentence)
        merged_sentences = []
        for sentence in sentences:
            if len(sentence) < 10 and merged_sentences:
                merged_sentences[-1] += " " + sentence
            else:
                merged_sentences.append(sentence)
        
        return merged_sentences
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def split_into_characters(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into character chunks (useful for very long texts)"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            # Try to break at a sentence boundary if possible
            if i + chunk_size < len(text):
                last_period = chunk.rfind('.')
                last_question = chunk.rfind('?')
                last_exclamation = chunk.rfind('!')
                last_boundary = max(last_period, last_question, last_exclamation)
                if last_boundary > chunk_size // 2:
                    chunk = text[i:i+last_boundary+1]
                    chunks.append(chunk.strip())
                    # Adjust i for next iteration
                    i += last_boundary - chunk_size
                    continue
            
            chunks.append(chunk.strip())
        
        return [c for c in chunks if c]
    
    def generate_audio_for_text(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Generate audio for a block of text"""
        try:
            audio_chunks = []
            generator = self.pipeline(
                text,
                voice=self.voice,
                speed=speed,
                split_pattern=r'\n+'
            )
            
            for gs, ps, audio in generator:
                if audio is not None and len(audio) > 0:
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    if len(audio.shape) > 1:
                        audio = audio.squeeze()
                    audio_chunks.append(audio)
            
            if not audio_chunks:
                return np.array([])
            
            return np.concatenate(audio_chunks)
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return np.array([])
    
    async def generate_audio_stream(self, request: StreamingChapterRequest) -> AsyncGenerator[bytes, None]:
        """Generate audio as a stream with chunking"""
        try:
            # Split content based on chunking strategy
            if request.chunk_size == 'sentence':
                chunks = self.split_into_sentences(request.content)
            elif request.chunk_size == 'paragraph':
                chunks = self.split_into_paragraphs(request.content)
            else:  # character
                chunks = self.split_into_characters(request.content, chunk_size=500)
            
            logger.info(f"Streaming {len(chunks)} chunks for chapter '{request.title or 'Untitled'}'")
            
            # Store original voice
            original_voice = self.voice
            self.voice = request.voice
            
            try:
                total_duration = 0
                start_time = time.time()
                
                # Send metadata first if requested
                if request.stream_metadata:
                    metadata = {
                        "type": "metadata",
                        "title": request.title,
                        "total_chunks": len(chunks),
                        "voice": request.voice,
                        "speed": request.speed,
                        "sample_rate": self.sample_rate,
                        "chunk_size": request.chunk_size,
                        "timestamp": start_time
                    }
                    yield f"data: {json.dumps(metadata)}\n\n".encode('utf-8')
                
                # Process each chunk
                for i, chunk_text in enumerate(chunks, 1):
                    chunk_start_time = time.time()
                    
                    # Generate audio for this chunk
                    audio = self.generate_audio_for_text(chunk_text, request.speed)
                    
                    if len(audio) > 0:
                        chunk_duration = len(audio) / self.sample_rate
                        total_duration += chunk_duration
                        
                        # Convert audio to base64 for streaming
                        import base64
                        audio_bytes = BytesIO()
                        sf.write(audio_bytes, audio, self.sample_rate, format='WAV')
                        audio_bytes.seek(0)
                        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
                        
                        # Send audio chunk
                        audio_chunk = {
                            "type": "audio",
                            "index": i,
                            "text": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                            "duration": chunk_duration,
                            "audio": audio_base64,
                            "timestamp": time.time()
                        }
                        yield f"data: {json.dumps(audio_chunk)}\n\n".encode('utf-8')
                        
                        # Add silence between chunks if requested
                        if request.add_silence_between_chunks and i < len(chunks):
                            silence_duration = int((request.silence_duration_ms / 1000.0) * self.sample_rate)
                            if silence_duration > 0:
                                silence = np.zeros(silence_duration)
                                silence_audio = BytesIO()
                                sf.write(silence_audio, silence, self.sample_rate, format='WAV')
                                silence_audio.seek(0)
                                silence_base64 = base64.b64encode(silence_audio.read()).decode('utf-8')
                                
                                silence_chunk = {
                                    "type": "silence",
                                    "duration_ms": request.silence_duration_ms,
                                    "audio": silence_base64
                                }
                                yield f"data: {json.dumps(silence_chunk)}\n\n".encode('utf-8')
                    
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)
                
                # Send completion message
                completion = {
                    "type": "complete",
                    "total_duration": total_duration,
                    "total_chunks": len(chunks),
                    "total_time": time.time() - start_time,
                    "is_final": True
                }
                yield f"data: {json.dumps(completion)}\n\n".encode('utf-8')
                
            finally:
                # Restore original voice
                self.voice = original_voice
                
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            error_msg = {
                "type": "error",
                "error": str(e),
                "is_final": True
            }
            yield f"data: {json.dumps(error_msg)}\n\n".encode('utf-8')
    
    def generate_chapter_audio(self, chapter: ChapterRequest) -> Tuple[np.ndarray, int]:
        """Generate audio for a complete chapter with sentence chunking"""
        try:
            # Split into sentences
            sentences = self.split_into_sentences(chapter.content)
            logger.info(f"Chapter '{chapter.title or 'Untitled'}': {len(sentences)} sentences")
            
            # Generate audio for each sentence
            audio_segments = []
            for i, sentence in enumerate(sentences, 1):
                if i % 10 == 0:
                    logger.debug(f"  Processed {i}/{len(sentences)} sentences")
                
                audio = self.generate_audio_for_text(sentence, chapter.speed)
                if len(audio) > 0:
                    audio_segments.append(audio)
            
            if not audio_segments:
                raise ValueError("No audio generated for chapter")
            
            # Add silence between sentences if requested
            if chapter.add_silence_between_sentences:
                silence_duration = int((chapter.silence_duration_ms / 1000.0) * self.sample_rate)
                silence = np.zeros(silence_duration)
                
                final_audio_with_pauses = []
                for i, segment in enumerate(audio_segments):
                    final_audio_with_pauses.append(segment)
                    if i < len(audio_segments) - 1:
                        final_audio_with_pauses.append(silence)
                
                final_audio = np.concatenate(final_audio_with_pauses)
            else:
                final_audio = np.concatenate(audio_segments)
            
            # Normalize audio
            max_val = np.max(np.abs(final_audio))
            if max_val > 1.0:
                final_audio = final_audio / max_val
            
            return final_audio, len(sentences)
            
        except Exception as e:
            logger.error(f"Error generating chapter audio: {e}")
            raise
    
    def generate_book_audio(self, book_request: BookRequest) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate audio for a complete book with multiple chapters"""
        try:
            all_audio = []
            chapter_info = []
            total_sentences = 0
            
            # Store original voice
            original_voice = self.voice
            self.voice = book_request.voice
            
            try:
                for i, chapter in enumerate(book_request.chapters, 1):
                    logger.info(f"Processing chapter {i}/{len(book_request.chapters)}: {chapter.title or f'Chapter {i}'}")
                    
                    # Use chapter-specific settings, fallback to book defaults
                    voice = chapter.voice if chapter.voice != "am_adam" else book_request.voice
                    if voice != self.voice:
                        self.voice = voice
                    
                    # Generate chapter audio
                    chapter_audio, sentence_count = self.generate_chapter_audio(chapter)
                    
                    if len(chapter_audio) > 0:
                        all_audio.append(chapter_audio)
                        total_sentences += sentence_count
                        chapter_info.append({
                            "index": i,
                            "title": chapter.title or f"Chapter {i}",
                            "duration": len(chapter_audio) / self.sample_rate,
                            "sentences": sentence_count
                        })
                        
                        # Add silence between chapters if requested
                        if book_request.add_chapter_markers and i < len(book_request.chapters):
                            # Add 2 seconds of silence between chapters
                            chapter_silence = np.zeros(int(2.0 * self.sample_rate))
                            all_audio.append(chapter_silence)
                
                if not all_audio:
                    raise ValueError("No audio generated for any chapter")
                
                # Concatenate all audio
                final_audio = np.concatenate(all_audio)
                
                # Final normalization
                max_val = np.max(np.abs(final_audio))
                if max_val > 1.0:
                    final_audio = final_audio / max_val
                
                total_duration = len(final_audio) / self.sample_rate
                
                metadata = {
                    "title": book_request.title,
                    "chapters": chapter_info,
                    "total_duration": total_duration,
                    "total_sentences": total_sentences,
                    "total_chapters": len(book_request.chapters)
                }
                
                return final_audio, metadata
                
            finally:
                # Restore original voice
                self.voice = original_voice
                
        except Exception as e:
            logger.error(f"Error generating book audio: {e}")
            raise
    
    def save_as_mp3(self, audio: np.ndarray, output_path: Path) -> Path:
        """Save audio as MP3 using ffmpeg, fallback to WAV"""
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_wav = Path(tmp_file.name)
        
        try:
            # Save as WAV first
            sf.write(temp_wav, audio, self.sample_rate)
            
            # Try to convert to MP3 with ffmpeg
            try:
                # Check if ffmpeg is available
                subprocess.run(['ffmpeg', '-version'], 
                             capture_output=True, check=True)
                
                # Convert to MP3
                cmd = [
                    'ffmpeg',
                    '-i', str(temp_wav),
                    '-codec:a', 'libmp3lame',
                    '-qscale:a', '2',  # Quality (0-9, 2 is good)
                    '-y',  # Overwrite output
                    str(output_path)
                ]
                
                subprocess.run(cmd, capture_output=True, check=True)
                logger.info(f"Saved as MP3: {output_path}")
                return output_path
                
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(f"FFmpeg not available, saving as WAV: {e}")
                wav_output = output_path.with_suffix('.wav')
                shutil.copy2(temp_wav, wav_output)
                logger.info(f"Saved as WAV instead: {wav_output}")
                return wav_output
                
        finally:
            # Clean up temp file
            if temp_wav.exists():
                temp_wav.unlink()
    
    @staticmethod
    def list_voices() -> List[str]:
        """List available voices"""
        return [
            'af_bella', 'af_nicole', 'af_sarah', 'af_sky',
            'am_adam', 'am_michael', 'am_leo',
            'bf_emma', 'bf_isabella', 'bf_olivia',
            'bm_george', 'bm_lewis', 'bm_daniel'
        ]


# FastAPI application
app = FastAPI(
    title="Kokoro TTS API - Chapter Mode",
    description="Text-to-Speech API with advanced chapter support for books and long texts",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global TTS instance
tts_instance: Optional[KokoroTTS] = None

@app.on_event("startup")
async def startup_event():
    """Initialize TTS on startup"""
    global tts_instance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        tts_instance = KokoroTTS(device=device)
        logger.info(f"TTS initialized successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {e}")
        tts_instance = None

async def delayed_cleanup(file_path: Path, delay_seconds: int = 60):
    """Clean up file after a delay"""
    await asyncio.sleep(delay_seconds)
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up {file_path}: {e}")

def cleanup_file(file_path: Path):
    """Clean up temporary files immediately"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up {file_path}: {e}")

@app.get("/", response_model=Dict[str, Any])
async def root():
    """API information"""
    return {
        "name": "Kokoro TTS API - Chapter Mode",
        "version": "2.0.0",
        "status": "running" if tts_instance else "initializing",
        "device": tts_instance.device if tts_instance else "unknown",
        "sample_rate": tts_instance.sample_rate if tts_instance else 0,
        "endpoints": [
            "/synthesize/chapter - Convert a single chapter",
            "/synthesize/stream - Stream chapter audio in real-time",
            "/synthesize/book - Convert a complete book with multiple chapters",
            "/synthesize/file - Convert from text file",
            "/voices - List available voices",
            "/download/{file_id} - Download generated audio"
        ]
    }

@app.get("/voices", response_model=VoicesResponse)
async def get_voices():
    """List available voices"""
    return {
        "voices": KokoroTTS.list_voices(),
        "default": "am_adam",
        "recommended_for_narration": ["am_adam", "am_michael", "am_leo"]
    }

@app.post("/synthesize/chapter")
async def synthesize_chapter(
    chapter: ChapterRequest,
    background_tasks: BackgroundTasks,
    return_audio: bool = False
):
    """
    Convert a single chapter to speech (MP3 format)
    
    - **title**: Optional chapter title
    - **content**: Chapter text content
    - **voice**: Voice to use (default: am_adam)
    - **speed**: Speech speed (0.5-2.0, default: 1.0)
    - **add_silence_between_sentences**: Add silence between sentences (default: true)
    - **silence_duration_ms**: Silence duration in milliseconds (default: 100)
    - **return_audio**: If true, returns the audio file directly instead of JSON
    """
    if not tts_instance:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    try:
        # Store original voice
        original_voice = tts_instance.voice
        tts_instance.voice = chapter.voice
        
        try:
            # Generate audio
            audio, sentence_count = tts_instance.generate_chapter_audio(chapter)
            
            # Generate unique filename
            file_id = uuid.uuid4().hex
            title_slug = re.sub(r'[^a-zA-Z0-9]', '_', chapter.title or 'chapter')[:50]
            filename = f"{title_slug}_{file_id}.mp3"
            output_path = OUTPUT_DIR / filename
            
            # Save as MP3
            tts_instance.save_as_mp3(audio, output_path)
            
            # Calculate duration
            duration = len(audio) / tts_instance.sample_rate
            
            # Return audio file directly if requested
            if return_audio:
                # Don't cleanup immediately - let the file be served first
                # Schedule cleanup after 5 minutes instead
                background_tasks.add_task(delayed_cleanup, output_path, 300)
                
                return FileResponse(
                    path=output_path,
                    media_type="audio/mpeg",
                    filename=filename
                )
            
            # Otherwise return JSON response with file info and schedule cleanup
            background_tasks.add_task(cleanup_file, output_path)
            
            return TTSResponse(
                success=True,
                file_id=file_id,
                filename=filename,
                duration=duration,
                sentences=sentence_count,
                message=f"Chapter processed successfully: {sentence_count} sentences, {duration:.2f} seconds"
            )
            
        finally:
            # Restore original voice
            tts_instance.voice = original_voice
            
    except Exception as e:
        logger.error(f"Chapter synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize/stream")
async def synthesize_stream(
    request: StreamingChapterRequest
):
    """
    Stream chapter audio in real-time with server-sent events
    
    This endpoint streams audio chunks as they're generated, allowing for
    real-time playback without waiting for the entire synthesis to complete.
    
    - **title**: Optional chapter title
    - **content**: Chapter text content
    - **voice**: Voice to use (default: am_adam)
    - **speed**: Speech speed (0.5-2.0, default: 1.0)
    - **chunk_size**: Chunking strategy - 'sentence', 'paragraph', or 'character'
    - **add_silence_between_chunks**: Add silence between chunks (default: true)
    - **silence_duration_ms**: Silence duration in milliseconds (default: 100)
    - **stream_metadata**: Include metadata in the stream (default: true)
    
    Returns a Server-Sent Events (SSE) stream with:
    - Metadata chunk: Information about the stream
    - Audio chunks: Base64-encoded WAV audio chunks
    - Silence chunks: Silence between audio chunks
    - Complete chunk: Final status with total duration
    """
    if not tts_instance:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    async def generate():
        try:
            async for chunk in tts_instance.generate_audio_stream(request):
                yield chunk
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            error_msg = {
                "type": "error",
                "error": str(e),
                "is_final": True
            }
            yield f"data: {json.dumps(error_msg)}\n\n".encode('utf-8')
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@app.post("/synthesize/book")
async def synthesize_book(
    book: BookRequest,
    background_tasks: BackgroundTasks,
    return_audio: bool = False
):
    """
    Convert a complete book with multiple chapters to speech (MP3 format)
    
    - **title**: Book title
    - **chapters**: List of chapters with content
    - **voice**: Default voice for all chapters
    - **speed**: Default speech speed
    - **add_chapter_markers**: Add audio markers between chapters (default: true)
    - **return_audio**: If true, returns the audio file directly instead of JSON
    """
    if not tts_instance:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    try:
        # Generate complete book audio
        audio, metadata = tts_instance.generate_book_audio(book)
        
        # Generate unique filename
        file_id = uuid.uuid4().hex
        title_slug = re.sub(r'[^a-zA-Z0-9]', '_', book.title)[:50]
        filename = f"{title_slug}_{file_id}.mp3"
        output_path = OUTPUT_DIR / filename
        
        # Save as MP3
        tts_instance.save_as_mp3(audio, output_path)
        
        # Return audio file directly if requested
        if return_audio:
            # Schedule cleanup after 10 minutes for longer books
            background_tasks.add_task(delayed_cleanup, output_path, 600)
            
            return FileResponse(
                path=output_path,
                media_type="audio/mpeg",
                filename=filename
            )
        
        # Otherwise return JSON response with file info and schedule cleanup
        background_tasks.add_task(cleanup_file, output_path)
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "metadata": metadata,
            "message": f"Book processed successfully: {metadata['total_chapters']} chapters, "
                      f"{metadata['total_sentences']} sentences, "
                      f"{metadata['total_duration']:.2f} seconds"
        }
        
    except Exception as e:
        logger.error(f"Book synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize/file")
async def synthesize_from_file(
    file: UploadFile = File(..., description="Text file to convert"),
    voice: str = Form("am_adam", description="Voice to use"),
    speed: float = Form(1.0, ge=0.5, le=2.0, description="Speech speed"),
    title: Optional[str] = Form(None, description="Chapter title"),
    add_silence_between_sentences: bool = Form(True, description="Add silence between sentences"),
    silence_duration_ms: int = Form(100, ge=0, le=1000, description="Silence duration in milliseconds"),
    background_tasks: BackgroundTasks = None
):
    """
    Convert text from a file to speech (MP3 format)
    
    Upload a text file and convert it to speech
    """
    if not tts_instance:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8').strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Create chapter request
        chapter = ChapterRequest(
            title=title or file.filename,
            content=text,
            voice=voice,
            speed=speed,
            add_silence_between_sentences=add_silence_between_sentences,
            silence_duration_ms=silence_duration_ms
        )
        
        # Store original voice
        original_voice = tts_instance.voice
        tts_instance.voice = voice
        
        try:
            # Generate audio
            audio, sentence_count = tts_instance.generate_chapter_audio(chapter)
            
            # Generate unique filename
            file_id = uuid.uuid4().hex
            filename = f"{file_id}.mp3"
            output_path = OUTPUT_DIR / filename
            
            # Save as MP3
            tts_instance.save_as_mp3(audio, output_path)
            
            # Schedule cleanup after 5 minutes
            if background_tasks:
                background_tasks.add_task(delayed_cleanup, output_path, 300)
            
            return FileResponse(
                path=output_path,
                media_type="audio/mpeg",
                filename=f"{file.filename.rsplit('.', 1)[0]}.mp3"
            )
            
        finally:
            tts_instance.voice = original_voice
            
    except Exception as e:
        logger.error(f"File synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_id}")
async def download_audio(file_id: str, background_tasks: BackgroundTasks):
    """
    Download generated audio file
    
    - **file_id**: ID of the generated audio file
    """
    # Check for mp3 files
    for ext in ['mp3', 'wav']:
        file_path = OUTPUT_DIR / f"{file_id}.{ext}"
        if file_path.exists():
            # Schedule cleanup after download
            background_tasks.add_task(delayed_cleanup, file_path, 60)
            return FileResponse(
                path=file_path,
                media_type=f"audio/{ext}",
                filename=f"speech.{ext}"
            )
    
    # Also check for files with custom names (containing file_id)
    for file_path in OUTPUT_DIR.glob(f"*{file_id}*.mp3"):
        if file_path.exists():
            background_tasks.add_task(delayed_cleanup, file_path, 60)
            return FileResponse(
                path=file_path,
                media_type="audio/mpeg",
                filename=file_path.name
            )
    
    raise HTTPException(status_code=404, detail="File not found")

@app.delete("/cleanup")
async def cleanup_old_files(hours: int = 24):
    """Clean up old generated files"""
    try:
        import time
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        cleaned = 0
        for file_path in OUTPUT_DIR.glob("*"):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned += 1
        
        return {"message": f"Cleaned up {cleaned} files older than {hours} hours"}
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )