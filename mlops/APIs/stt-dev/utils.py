import os
import logging
import subprocess
import io
import uuid
import librosa
from inference import transcribe_audio
from config import Config
from models import TranscriptionResponse
from pydub import AudioSegment
from fastapi import HTTPException, status, UploadFile, Header
from typing import List, Dict
import soundfile as sf


logger = logging.getLogger(__name__)
AUDIO_SAMPLE_RATE = 16000  # Fixed 16kHz for model inference


def process_video(file: UploadFile, news_type: str = None, language: str = None):
    try:
        file_content = file.file.read()
        file_extension = os.path.splitext(file.filename)[1].lower()
        original_file_name = f"{uuid.uuid4()}{file_extension}"
        tmp_file_path = os.path.join(Config.UPLOADS_DIR, original_file_name)

        with open(tmp_file_path, "wb") as f:
            f.write(file_content)

        if file_extension in Config.VIDEO_EXTENSIONS:
            audio_file_path = os.path.join(Config.UPLOADS_DIR, f"{original_file_name}.wav")
            try:
                extract_audio_from_video(tmp_file_path, audio_file_path)
            except HTTPException as e:
                # Re-raise specific HTTP exceptions from audio extraction
                raise e
            
            if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
                # This case might be redundant if extract_audio_from_video handles it, but it's a good safeguard.
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No audio found in the provided video.")
        else:
            audio_file_path = os.path.join(Config.UPLOADS_DIR, f"{original_file_name}.wav")
            librosa_audio, sr = librosa.load(io.BytesIO(file_content), sr=AUDIO_SAMPLE_RATE)
            sf.write(audio_file_path, librosa_audio, sr, format='WAV')
            # audio_file_path = os.path.join(Config.UPLOADS_DIR, f"{original_file_name}.wav")
            # librosa_audio, sr = librosa.load(io.BytesIO(file_content), sr=AUDIO_SAMPLE_RATE)
            # librosa.output.write_wav(audio_file_path, librosa_audio, sr)

        all_chunks = split_audio_into_chunks(audio_path=audio_file_path,
                                             output_folder=Config.UPLOADS_DIR)

        full_text = ""
        segment_info = []

        for chunk in all_chunks:
            try:
                start_time = chunk["start"]
                end_time = chunk["end"]
                chunk_path = chunk["chunk_path"]

                text = transcribe_audio(chunk_path)

                if full_text:
                    last_word = full_text.split()[-1]
                    first_word = text.split()[0]
                    if last_word == first_word:
                        text = " ".join(text.split()[1:])
                full_text = (full_text + " " + text).strip()

                segment_info.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text
                })

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_path}: {str(e)}")
                continue

        # Delete uploaded file and audio chunks
        os.remove(tmp_file_path)
        os.remove(audio_file_path)
        for chunk in all_chunks:
            if os.path.exists(chunk["chunk_path"]):
                os.remove(chunk["chunk_path"])

        return TranscriptionResponse(full_text=full_text, timestamp=segment_info)

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Error {str(e)}")


def validate_api_key(api_key: str = Header(None, alias="X-API-Key")):
    keys = Config.USER_API_KEYS
    if api_key not in keys.values():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def validate_language(language: str = Header(..., alias="X-Language")):
    language = language.lower()
    if language not in Config.SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    return language


def validate_news_type(news_type: str = Header(..., alias="X-News-Type")):
    news_type = news_type.lower()
    if news_type not in Config.SUPPORTED_NEWS_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported news type: {news_type}")
    return news_type


def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    try:
        command = [
            'ffmpeg',
            '-i', video_path,           # Input video
            '-q:a', '0',               # Highest audio quality
            '-map', 'a',               # Map only audio stream
            '-ar', str(AUDIO_SAMPLE_RATE),            # Set sample rate to 16kHz
            '-ac', '1',                # Convert to mono
            '-y',                      # Overwrite output without prompt
            output_audio_path
        ]
        
        logger.debug(f"Extracting audio: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        
        # Verify extraction was successful
        if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No audio track found in video file"
            )
            
        logger.info(f"Audio extracted successfully: {output_audio_path}")
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        
        # Check for specific ffmpeg error indicating no audio stream
        if "Stream map 'a' matches no streams" in error_msg:
            logger.warning(f"No audio stream found in {video_path}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No audio stream found in the video file."
            )
        
        logger.error(f"FFmpeg error extracting audio from {video_path}: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract audio from video: {error_msg}"
        )
    except FileNotFoundError:
        logger.error("FFmpeg not found in system PATH")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FFmpeg is required but not installed"
        )



def split_audio_into_chunks(
    audio_path: str,
    output_folder: str,
    chunk_duration_ms: int = 10000,  # 10 seconds
    min_chunk_duration_ms: int = 1000  # 1 second
) -> List[Dict[str, any]]:
    try:
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Load audio file
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid or corrupted audio file: {str(e)}"
            )
        
        total_duration_ms = len(audio)
        logger.info(f"Splitting audio: {total_duration_ms/1000:.2f}s into {chunk_duration_ms/1000}s chunks")
        
        chunks_info = []
        chunk_number = 1
        previous_chunk = None
        previous_start_ms = 0
        
        # Process audio in chunks
        for start_ms in range(0, total_duration_ms, chunk_duration_ms):
            end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
            current_chunk = audio[start_ms:end_ms]
            
            # Handle short final chunks by merging with previous chunk
            if len(current_chunk) < min_chunk_duration_ms and end_ms == total_duration_ms:
                if previous_chunk is not None and chunks_info:
                    # Merge with previous chunk
                    merged_chunk = previous_chunk + current_chunk
                    
                    # Update previous chunk file
                    previous_chunk_path = chunks_info[-1]["chunk_path"]
                    try:
                        merged_chunk.export(previous_chunk_path, format="wav")
                        
                        # Update the end time of the merged chunk
                        chunks_info[-1]["end"] = end_ms / 1000.0
                        
                        logger.debug(f"Merged final short chunk with previous chunk")
                        
                    except Exception as e:
                        logger.error(f"Failed to export merged chunk: {e}")
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to process audio chunks"
                        )
                break  # Don't process this chunk separately
            
            # Save chunk as separate file
            chunk_filename = os.path.join(output_folder, f"chunk_{chunk_number:04d}.wav")
            
            try:
                current_chunk.export(chunk_filename, format="wav")
                
                # Convert times to seconds
                start_time_sec = start_ms / 1000.0
                end_time_sec = end_ms / 1000.0
                
                chunks_info.append({
                    "start": start_time_sec,
                    "end": end_time_sec,
                    "chunk_path": chunk_filename
                })
                
                # Store for potential merging with next chunk
                previous_chunk = current_chunk
                previous_start_ms = start_ms
                
                chunk_number += 1
                
                logger.debug(f"Created chunk {chunk_number-1}: {start_time_sec:.2f}s - {end_time_sec:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to export chunk {chunk_number}: {e}")
                continue  # Skip this chunk and continue with next
        
        logger.info(f"Audio split into {len(chunks_info)} chunks in {output_folder}")
        return chunks_info
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in audio chunking: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {str(e)}"
        )