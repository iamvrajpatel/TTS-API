import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment


def save_as_mp3(audio_array: np.ndarray, output_path: str, sample_rate: int = 24000) -> str:
    """Convert numpy array to MP3 file and return the path"""
    wav_path = output_path.replace('.mp3', '_temp.wav')
    mp3_path = output_path
    
    # Save as WAV first
    sf.write(wav_path, audio_array, sample_rate)
    
    # Convert to MP3
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format='mp3')
    
    # Clean up temporary WAV file
    os.remove(wav_path)
    return mp3_path