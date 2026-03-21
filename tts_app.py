#!/usr/bin/env python3
"""
Kokoro TTS Application - Corrected version with proper tensor handling
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
    """Kokoro TTS wrapper class with proper tensor handling"""
    
    def __init__(self, device='cpu', voice='af_bella', repo_id='hexgrad/Kokoro-82M'):
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
    
    def text_to_speech(self, text, output_file, speed=1.0):
        """
        Convert text to speech and save as MP3
        
        Args:
            text: Input text to convert
            output_file: Output file path
            speed: Speech speed (0.5 to 2.0)
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
            try:
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
                
        except Exception as e:
            print(f"Error generating speech: {e}", file=sys.stderr)
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
                result = subprocess.run(cmd, capture_output=True, check=True)
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
        description='Kokoro TTS - Convert text to speech',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello world" -o hello.mp3
  %(prog)s -f input.txt -o output.mp3
  %(prog)s -i --voice af_nicole --speed 1.2
  %(prog)s --list-voices
        """
    )
    
    parser.add_argument(
        'text',
        nargs='?',
        help='Text to convert to speech'
    )
    
    parser.add_argument(
        '-f', '--file',
        help='Input text file (alternative to providing text directly)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='output.mp3',
        help='Output file path (default: output.mp3)'
    )
    
    parser.add_argument(
        '-v', '--voice',
        default='af_bella',
        help='Voice to use (default: af_bella)'
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
        '--list-voices',
        action='store_true',
        help='List available voices and exit'
    )
    
    args = parser.parse_args()
    
    # List voices if requested
    if args.list_voices:
        tts = KokoroTTS(device='cpu')
        print("Available voices:")
        for voice in tts.list_voices():
            print(f"  {voice}")
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
        print("Enter sentences to convert (type 'quit' to exit)")
        print("-"*60)
        
        while True:
            try:
                text = input("\n📝 Enter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if text:
                    import time
                    timestamp = int(time.time())
                    output_file = f"output_{timestamp}.mp3"
                    try:
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
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                print("File is empty", file=sys.stderr)
                sys.exit(1)
            
            tts.text_to_speech(text, args.output, speed=args.speed)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    
    elif args.text:
        tts.text_to_speech(args.text, args.output, speed=args.speed)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
