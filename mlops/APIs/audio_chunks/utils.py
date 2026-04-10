from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import traceback
import subprocess
import logging
import os

logger = logging.getLogger(__name__)



def extract_audio(video_path: str, audio_path: str):
    """
    Extract audio from video to WAV using ffmpeg (optimized).
    Handles minor stream corruption gracefully.
    """
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-fflags", "+genpts+discardcorrupt",
        "-err_detect", "ignore_err",
        "-i", video_path,
        "-vn",
        "-map", "0:a:0?",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-threads", "2",  # Use 2 threads for faster encoding
        audio_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        return audio_path
    except subprocess.TimeoutExpired:
        logger.error("Audio extraction timeout (300s)")
        raise RuntimeError("Audio extraction timeout")
    except subprocess.CalledProcessError as e:
        # Accept partial audio if extraction was partially successful
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 16000 * 2:
            logger.warning("Partial audio extracted, proceeding...")
            return audio_path
        raise RuntimeError("Failed to extract audio from video")

def is_audio_file(file_path: str) -> bool:
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=nw=1:nk=1",
        file_path
    ]
    r = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode == 0 and r.stdout.strip() == b"audio"

def get_large_audio_chunks_on_silence(path, debug=False):
    try:
        # Load audio once with optimized parameters
        sound = AudioSegment.from_file(path)
        sound = effects.normalize(sound)  # Normalize audio levels
        sound = sound.high_pass_filter(100)  # Remove low frequency noise
        
        # Split audio with optimized parameters
        chunks = split_on_silence(
            sound,
            min_silence_len=1200,  # 1.2 seconds for better speaker separation
            silence_thresh=sound.dBFS - 18,  # Balanced threshold for mixed speech
            keep_silence=600,  # 0.6 seconds to maintain context
        )
        
        # Fast filtering using list comprehension with single pass
        min_chunk_length = 2000  # Minimum 2 seconds for valid speech
        chunk_times = []
        start_time = 0
        
        for chunk in chunks:
            chunk_length = len(chunk)
            if chunk_length >= min_chunk_length:
                chunk_times.append((start_time, start_time + chunk_length))
            start_time += chunk_length
            
        return chunk_times
        
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error processing audio: {str(e)}")
        return []
