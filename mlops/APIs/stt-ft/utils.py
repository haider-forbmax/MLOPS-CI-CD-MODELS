import os
import logging
import subprocess
import io
import uuid
import librosa
import traceback
from inference import transcribe_audio
from config import Config
from models import TranscriptionResponse, Segment  # Import both models
from pydub import AudioSegment
from fastapi import HTTPException, status, UploadFile, Header
from typing import List, Dict
import soundfile as sf


logger = logging.getLogger(__name__)
AUDIO_SAMPLE_RATE = 16000  # Fixed 16kHz for model inference


def process_video(file: UploadFile, news_type: str = None, language: str = None) -> TranscriptionResponse:
    audio_file_path = None
    tmp_file_path = None
    
    try:
        # Read file content
        file_content = file.file.read()
        file_extension = os.path.splitext(file.filename)[1].lower()
        original_file_name = f"{uuid.uuid4()}{file_extension}"
        tmp_file_path = os.path.join(Config.UPLOADS_DIR, original_file_name)

        logger.info(f"Processing file: {file.filename} ({len(file_content)} bytes)")
        logger.info(f"Temp file path: {tmp_file_path}")

        # Save uploaded file
        with open(tmp_file_path, "wb") as f:
            f.write(file_content)
        
        if not os.path.exists(tmp_file_path):
            raise HTTPException(
                status_code=500,
                detail="Failed to save uploaded file"
            )
        
        logger.info(f"File saved successfully: {tmp_file_path}")

        # --- Step 1: Convert to WAV ---
        audio_file_path = os.path.join(Config.UPLOADS_DIR, f"{uuid.uuid4()}.wav")
        
        if file_extension in Config.VIDEO_EXTENSIONS:
            logger.info(f"Extracting audio from video: {file_extension}")
            try:
                extract_audio_from_video(tmp_file_path, audio_file_path)
            except HTTPException as e:
                traceback.print_exc()
                raise e

            if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No audio found in the provided video."
                )
            logger.info(f"Audio extracted: {audio_file_path} ({os.path.getsize(audio_file_path)} bytes)")
        else:
            logger.info(f"Converting audio file: {file_extension}")
            try:
                convert_to_wav_ffmpeg(
                    input_path=tmp_file_path,
                    output_path=audio_file_path,
                    sample_rate=AUDIO_SAMPLE_RATE
                )
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Audio normalization failed: {str(e)}"
                )

            if not os.path.exists(audio_file_path) or os.path.getsize(audio_file_path) == 0:
                raise HTTPException(
                    status_code=422,
                    detail="No audio stream found in the uploaded file."
                )

            logger.info(f"Audio ready: {audio_file_path} ({os.path.getsize(audio_file_path)} bytes)")

        # Verify audio file exists before transcription
        if not os.path.exists(audio_file_path):
            raise HTTPException(
                status_code=500,
                detail=f"Audio file not found at: {audio_file_path}"
            )
        
        logger.info(f"Starting transcription for: {audio_file_path}")

        # --- Step 2: Transcribe full audio file ---
        try:
            transcription_result = transcribe_audio(audio_file_path)
        except HTTPException as e:
            traceback.print_exc()
            logger.error(f"Transcription HTTP error: {e.detail}")
            raise e
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Unexpected transcription error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Internal transcription error"
            )

        # --- Step 3: Process transcription result ---
        # Handle both old (string) and new (dict) response formats
        if isinstance(transcription_result, str):
            full_text = transcription_result.strip()
            segments = None
        elif isinstance(transcription_result, dict):
            full_text = transcription_result.get('text', '').strip()
            segments = transcription_result.get('segments', None)
        else:
            logger.error(f"Unexpected transcription result type: {type(transcription_result)}")
            full_text = str(transcription_result)
            segments = None

        # --- Step 4: Create timestamp info ---
        if segments and len(segments) > 0:
            segment_list = [
                Segment(
                    start_time=seg.get('start', 0.0),
                    end_time=seg.get('end', 0.0),
                    text=seg.get('text', '').strip()
                )
                for seg in segments
            ]
            logger.info(f"Transcription completed with {len(segment_list)} segments")
        else:
            # Fallback: single segment with full duration
            try:
                duration = librosa.get_duration(path=audio_file_path)
            except Exception:
                duration = 0.0
            
            segment_list = [
                Segment(
                    start_time=0.0,
                    end_time=duration,
                    text=full_text
                )
            ]
            logger.info("Transcription completed with single segment")

        # --- Step 5: Clean up ---
        try:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                logger.info(f"Cleaned up temp file: {tmp_file_path}")
            
            if audio_file_path and os.path.exists(audio_file_path):
                os.remove(audio_file_path)
                logger.info(f"Cleaned up audio file: {audio_file_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up files: {str(e)}")

        # Return TranscriptionResponse model
        return TranscriptionResponse(
            full_text=full_text,
            timestamp=segment_list
        )

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        # Clean up on error
        try:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            if audio_file_path and os.path.exists(audio_file_path):
                os.remove(audio_file_path)
        except:
            pass
        raise e
        
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error processing video: {str(e)}")
        # Clean up on error
        try:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            if audio_file_path and os.path.exists(audio_file_path):
                os.remove(audio_file_path)
        except:
            pass
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

def convert_to_wav_ffmpeg(input_path: str, output_path: str, sample_rate: int):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vn",                 # ignore video if present
        "-ac", "1",            # mono
        "-ar", str(sample_rate),
        "-f", "wav",
        output_path
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    """Extract audio from video file and convert to 16kHz mono WAV"""
    try:
        # Verify input file exists
        if not os.path.exists(video_path):
            logger.error(f"Input video file not found: {video_path}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Video file not found: {video_path}"
            )
        
        logger.info(f"Input video exists: {video_path} ({os.path.getsize(video_path)} bytes)")
        logger.info(f"Output audio path: {output_audio_path}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_audio_path)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        command = [
            'ffmpeg',
            '-i', video_path,                      # Input video
            '-vn',                                 # No video output
            '-acodec', 'pcm_s16le',               # PCM 16-bit encoding
            '-ar', str(AUDIO_SAMPLE_RATE),        # Set sample rate to 16kHz
            '-ac', '1',                            # Convert to mono
            '-f', 'wav',                           # Force WAV format
            '-y',                                  # Overwrite output without prompt
            output_audio_path
        ]
        
        logger.info(f"Extracting audio from: {video_path}")
        logger.debug(f"FFmpeg command: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        
        # Log FFmpeg output for debugging
        if result.stderr:
            logger.debug(f"FFmpeg stderr: {result.stderr[:500]}")  # First 500 chars
        
        # Verify extraction was successful
        if not os.path.exists(output_audio_path):
            logger.error(f"Output file was not created: {output_audio_path}")
            logger.error(f"FFmpeg stdout: {result.stdout}")
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Audio extraction failed - output file not created"
            )
        
        file_size = os.path.getsize(output_audio_path)
        if file_size == 0:
            logger.error(f"Output file is empty: {output_audio_path}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No audio track found in video file"
            )
            
        logger.info(f"✓ Audio extracted successfully: {output_audio_path} ({file_size:,} bytes)")
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        logger.error(f"FFmpeg process failed with return code {e.returncode}")
        logger.error(f"FFmpeg stderr: {error_msg}")
        
        # Check for specific ffmpeg errors
        if "Stream map" in error_msg or "does not contain any stream" in error_msg:
            logger.warning(f"No audio stream found in {video_path}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No audio stream found in the video file"
            )
        
        if "Invalid data found" in error_msg or "moov atom not found" in error_msg:
            logger.error(f"Corrupted or invalid video file: {video_path}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Video file is corrupted or invalid format"
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract audio: {error_msg[:200]}"
        )
        
    except FileNotFoundError:
        logger.error("FFmpeg not found in system PATH")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FFmpeg is required but not installed. Install with: apt-get install ffmpeg"
        )
    
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error during audio extraction: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio extraction failed: {str(e)}"
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