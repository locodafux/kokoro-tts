#!/usr/bin/env python3
"""Quick test for Kokoro TTS"""

import sys

def test_installation():
    """Test if all packages are installed correctly"""
    print("Testing installation...")
    
    # Test imports
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        
        import soundfile as sf
        print(f"✓ SoundFile version: {sf.__version__}")
        
        from kokoro import KPipeline
        print(f"✓ Kokoro imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    if test_installation():
        print("\n✓ All packages installed correctly!")
        print("\nNow run: python tts_app.py 'Hello world' -o hello.mp3")
    else:
        print("\n✗ Missing packages. Run: uv pip install kokoro torch soundfile numpy")
