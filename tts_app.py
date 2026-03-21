#!/usr/bin/env python3
"""
Kokoro TTS Application - Chapter Mode with Sentence Chunking
Handles long text by splitting into sentences, generating each, then merging
"""

import argparse
import sys
import warnings
import numpy as np
import soundfile as sf
from pathlib import Path
import subprocess
import tempfile
import torch
import re
from typing import List, Tuple
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import kokoro with error handling
try:
    from kokoro import KPipeline
except ImportError as e:
    print(f"Error importing kokoro: {e}")
    print("Please install kokoro with: uv pip install kokoro")
    sys.exit(1)


class KokoroTTS:
    """Kokoro TTS wrapper class with sentence chunking for long text"""
    
    def __init__(self, device='cpu', voice='am_adam', repo_id='hexgrad/Kokoro-82M'):
        """
        Initialize the TTS pipeline
        
        Args:
            device: 'cpu' or 'cuda' for GPU acceleration
            voice: Voice preset (af_bella, af_nicole, af_sarah, etc.)
            repo_id: Model repository ID
        """
        self.device = device
        self.voice = voice
        self.repo_id = repo_id
        
        # Check if CUDA is available
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        print(f"Initializing Kokoro TTS with device: {self.device}")
        print(f"Using voice: {voice}")
        
        try:
            # Initialize the pipeline
            self.pipeline = KPipeline(lang_code='a', device=self.device, repo_id=repo_id)
            
            # The sample rate is typically 24000 Hz for Kokoro
            self.sample_rate = 24000
            print(f"Using sample rate: {self.sample_rate} Hz")
            
        except Exception as e:
            print(f"Failed to initialize pipeline: {e}")
            raise
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Pattern to split sentences (handles common abbreviations)
        # Matches periods, question marks, exclamation points followed by space and capital letter
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        # Split into sentences
        sentences = re.split(sentence_pattern, text)
        
        # Clean up each sentence
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Merge very short sentences (less than 10 chars) with previous sentence
        merged_sentences = []
        for sentence in sentences:
            if len(sentence) < 10 and merged_sentences:
                merged_sentences[-1] += " " + sentence
            else:
                merged_sentences.append(sentence)
        
        return merged_sentences
    
    def generate_sentence_audio(self, sentence: str, speed: float = 1.0) -> np.ndarray:
        """
        Generate audio for a single sentence
        
        Args:
            sentence: Text to convert
            speed: Speech speed
            
        Returns:
            numpy array of audio samples
        """
        try:
            audio_chunks = []
            generator = self.pipeline(
                sentence,
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
            print(f"  ⚠️  Error generating sentence: {e}")
            return np.array([])
    
    def text_to_speech_chunked(self, text: str, output_file: str, speed: float = 1.0, 
                               show_progress: bool = True) -> str:
        """
        Convert long text to speech by splitting into sentences and merging
        
        Args:
            text: Input text (can be very long)
            output_file: Output file path
            speed: Speech speed
            show_progress: Show progress bar/text
            
        Returns:
            Output file path
        """
        try:
            # Validate input
            if not text or not text.strip():
                raise ValueError("Empty text provided")
            
            # Split into sentences
            print("\n📝 Splitting text into sentences...")
            sentences = self.split_into_sentences(text)
            print(f"✓ Split into {len(sentences)} sentences")
            
            # Generate audio for each sentence
            print("\n🎵 Generating audio for each sentence...")
            audio_segments = []
            total_sentences = len(sentences)
            
            for i, sentence in enumerate(sentences, 1):
                if show_progress:
                    # Show progress
                    progress = (i / total_sentences) * 100
                    bar_length = 40
                    filled = int(bar_length * i // total_sentences)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r  [{bar}] {i}/{total_sentences} ({progress:.1f}%) - {sentence[:50]}...", 
                          end='', flush=True)
                
                # Generate audio for this sentence
                audio = self.generate_sentence_audio(sentence, speed)
                
                if len(audio) > 0:
                    audio_segments.append(audio)
                
                # Small delay to prevent overwhelming the system
                if i % 10 == 0:
                    time.sleep(0.1)
            
            print()  # New line after progress
            
            if not audio_segments:
                raise ValueError("No audio generated for any sentence")
            
            # Merge all audio segments
            print("\n🔊 Merging audio segments...")
            final_audio = np.concatenate(audio_segments)
            
            # Add small pause between sentences? (optional)
            # Add 0.1 seconds of silence between sentences
            silence_duration = int(0.1 * self.sample_rate)
            silence = np.zeros(silence_duration)
            
            # Insert silence between segments
            final_audio_with_pauses = []
            for i, segment in enumerate(audio_segments):
                final_audio_with_pauses.append(segment)
                if i < len(audio_segments) - 1:  # Don't add silence after last segment
                    final_audio_with_pauses.append(silence)
            
            final_audio = np.concatenate(final_audio_with_pauses)
            
            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(final_audio))
            if max_val > 1.0:
                final_audio = final_audio / max_val
            
            total_duration = len(final_audio) / self.sample_rate
            print(f"✓ Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            
            # Save output
            print(f"\n💾 Saving to {output_file}...")
            output_path = Path(output_file)
            
            if output_path.suffix.lower() == '.mp3':
                self._save_as_mp3(final_audio, output_path)
            else:
                sf.write(output_path, final_audio, self.sample_rate)
                print(f"✓ Saved as WAV: {output_path}")
            
            print(f"✓ Successfully saved audio to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"\n❌ Error generating speech: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def text_to_speech(self, text: str, output_file: str, speed: float = 1.0) -> str:
        """
        Convert text to speech (single chunk, for shorter text)
        
        Args:
            text: Input text to convert
            output_file: Output file path
            speed: Speech speed
            
        Returns:
            Output file path
        """
        try:
            # Validate input
            if not text or not text.strip():
                raise ValueError("Empty text provided")
            
            print(f"Converting: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            print(f"Speed: {speed}x")
            
            # Generate speech audio
            audio_chunks = []
            
            # Process text through the pipeline
            generator = self.pipeline(
                text, 
                voice=self.voice, 
                speed=speed,
                split_pattern=r'\n+'
            )
            
            # Collect audio chunks
            chunk_count = 0
            for gs, ps, audio in generator:
                if audio is not None and len(audio) > 0:
                    # Convert tensor to numpy array if needed
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    
                    # Ensure it's a 1D array
                    if len(audio.shape) > 1:
                        audio = audio.squeeze()
                    
                    audio_chunks.append(audio)
                    chunk_count += 1
                    print(f"Generated chunk {chunk_count} (samples: {len(audio)})")
            
            if not audio_chunks:
                raise ValueError("No audio generated - pipeline returned empty result")
            
            # Concatenate all audio chunks
            final_audio = np.concatenate(audio_chunks)
            
            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(final_audio))
            if max_val > 1.0:
                final_audio = final_audio / max_val
            
            print(f"Total audio samples: {len(final_audio)}")
            print(f"Total audio duration: {len(final_audio) / self.sample_rate:.2f} seconds")
            
            # Determine output format
            output_path = Path(output_file)
            
            if output_path.suffix.lower() == '.mp3':
                # Save as MP3 using ffmpeg
                self._save_as_mp3(final_audio, output_path)
            else:
                # Save as WAV
                sf.write(output_path, final_audio, self.sample_rate)
                print(f"✓ Saved as WAV: {output_path}")
            
            print(f"✓ Successfully saved audio to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_as_mp3(self, audio, output_path):
        """Save audio as MP3 using ffmpeg"""
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
                
                # Run conversion
                subprocess.run(cmd, capture_output=True, check=True)
                print(f"✓ Converted to MP3 using ffmpeg")
                
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"FFmpeg not available or conversion failed: {e}")
                print("Saving as WAV instead")
                # Copy WAV to output path
                import shutil
                wav_output = output_path.with_suffix('.wav')
                shutil.copy2(temp_wav, wav_output)
                print(f"✓ Saved as WAV instead: {wav_output}")
                
        finally:
            # Clean up temp file
            if temp_wav.exists():
                temp_wav.unlink()
    
    def list_voices(self):
        """List available voices"""
        voices = [
            'af_bella', 'af_nicole', 'af_sarah', 'af_sky',
            'am_adam', 'am_michael', 'am_leo',
            'bf_emma', 'bf_isabella', 'bf_olivia',
            'bm_george', 'bm_lewis', 'bm_daniel'
        ]
        return voices


def main():
    """Main function to run the TTS application"""
    parser = argparse.ArgumentParser(
        description='Kokoro TTS - Convert text to speech with chapter/chunking support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Short text (standard mode)
  %(prog)s "Hello world" -o hello.mp3
  
  # Long text/chapter (automatic sentence chunking)
  %(prog)s -f chapter.txt -o audiobook.mp3 --chunk
  
  # Use custom voice and speed
  %(prog)s -f story.txt -o narration.mp3 -v am_adam -s 0.9 --chunk
  
  # Interactive mode
  %(prog)s -i --chunk
  
  # List available voices
  %(prog)s --list-voices
        """
    )
    
    parser.add_argument(
        'text',
        nargs='?',
        help='Text to convert to speech (short text)'
    )
    
    parser.add_argument(
        '-f', '--file',
        help='Input text file (supports long text/chapters)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='output.mp3',
        help='Output file path (default: output.mp3)'
    )
    
    parser.add_argument(
        '-v', '--voice',
        default='am_adam',
        help='Voice to use (default: am_adam) - best for narration'
    )
    
    parser.add_argument(
        '-s', '--speed',
        type=float,
        default=1.0,
        help='Speech speed (0.5 to 2.0, default: 1.0)'
    )
    
    parser.add_argument(
        '-d', '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for inference (default: cpu)'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--chunk',
        action='store_true',
        help='Enable sentence chunking for long text (recommended for chapters/books)'
    )
    
    parser.add_argument(
        '--list-voices',
        action='store_true',
        help='List available voices and exit'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )
    
    args = parser.parse_args()
    
    # List voices if requested
    if args.list_voices:
        tts = KokoroTTS(device='cpu')
        print("\nAvailable voices:")
        print("-" * 40)
        for voice in tts.list_voices():
            print(f"  {voice}")
        print("\nRecommended voices for narration:")
        print("  • am_adam - Best for professional narration")
        print("  • am_michael - Friendly, conversational")
        print("  • am_leo - Youthful, energetic")
        return
    
    # Initialize TTS
    try:
        tts = KokoroTTS(device=args.device, voice=args.voice)
    except Exception as e:
        print(f"Failed to initialize TTS: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get input text
    if args.interactive:
        print("\n" + "="*60)
        print("Kokoro TTS Interactive Mode")
        print("="*60)
        print(f"Voice: {args.voice} | Speed: {args.speed}x")
        if args.chunk:
            print("Mode: CHAPTER MODE (sentence chunking enabled)")
            print("  • Handles long text by splitting into sentences")
            print("  • Merges all sentences into one audio file")
        else:
            print("Mode: STANDARD MODE (best for short text)")
        print("\nEnter text (type 'quit' to exit, 'mode' to toggle chunking)")
        print("-"*60)
        
        chunk_mode = args.chunk
        
        while True:
            try:
                text = input("\n📝 Enter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if text.lower() == 'mode':
                    chunk_mode = not chunk_mode
                    print(f"✓ Chunking mode: {'ON' if chunk_mode else 'OFF'}")
                    continue
                if text:
                    import time
                    timestamp = int(time.time())
                    output_file = f"output_{timestamp}.mp3"
                    try:
                        if chunk_mode and len(text) > 200:
                            # Use chunked mode for longer text
                            tts.text_to_speech_chunked(
                                text, output_file, speed=args.speed,
                                show_progress=not args.no_progress
                            )
                        else:
                            tts.text_to_speech(text, output_file, speed=args.speed)
                        print(f"💾 Saved to: {output_file}")
                    except Exception as e:
                        print(f"❌ Error: {e}")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
    
    elif args.file:
        try:
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"File not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            
            print(f"\n📖 Reading file: {args.file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                print("File is empty", file=sys.stderr)
                sys.exit(1)
            
            # Show file statistics
            word_count = len(text.split())
            char_count = len(text)
            print(f"✓ File loaded: {word_count} words, {char_count} characters")
            
            # Choose mode based on text length and user preference
            use_chunking = args.chunk or len(text) > 500  # Auto-enable for long text
            
            if use_chunking:
                print("\n🎯 Using CHAPTER MODE (sentence chunking)")
                print("   This is recommended for long text to ensure stability")
                tts.text_to_speech_chunked(
                    text, args.output, speed=args.speed,
                    show_progress=not args.no_progress
                )
            else:
                print("\n🎯 Using STANDARD MODE (single chunk)")
                tts.text_to_speech(text, args.output, speed=args.speed)
                
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.text:
        # Use text from command line
        if args.chunk and len(args.text) > 200:
            tts.text_to_speech_chunked(
                args.text, args.output, speed=args.speed,
                show_progress=not args.no_progress
            )
        else:
            tts.text_to_speech(args.text, args.output, speed=args.speed)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()