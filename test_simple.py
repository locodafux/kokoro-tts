#!/usr/bin/env python3
"""Simple test for Kokoro TTS"""

import numpy as np
import soundfile as sf
import torch

def test_simple():
    """Simple test with minimal text"""
    try:
        from kokoro import KPipeline
        
        print("Initializing pipeline...")
        pipeline = KPipeline(lang_code='a')
        
        # Test with a very short sentence
        text = "Hello world"
        print(f"Testing with: {text}")
        
        # Generate audio
        generator = pipeline(text, voice='af_bella')
        
        # Get all audio chunks
        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(generator):
            if audio is not None:
                # Convert tensor to numpy if needed
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                    print(f"Chunk {i}: converted tensor to numpy, shape = {audio.shape}")
                else:
                    print(f"Chunk {i}: audio shape = {audio.shape if hasattr(audio, 'shape') else 'unknown'}")
                audio_chunks.append(audio)
        
        if audio_chunks:
            final_audio = np.concatenate(audio_chunks)
            print(f"Total audio samples: {len(final_audio)}")
            duration = len(final_audio) / 24000
            print(f"Duration: {duration:.2f} seconds")
            
            # Save as WAV
            sf.write('test_output.wav', final_audio, 24000)
            print("✓ Successfully saved test_output.wav")
            return True
        else:
            print("No audio chunks collected")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple()
