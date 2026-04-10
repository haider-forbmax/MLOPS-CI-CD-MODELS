import numpy as np
## import moviepy.editor as mp  # unused
## from fastapi.responses import JSONResponse  # unused
from utils import *
import os 
import shutil
import librosa
import tempfile
from config import Config
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput,  InferenceServerException
import traceback
import subprocess
## import time  # unused
import logging
logger = logging.getLogger(__name__)

class UserAudioError(Exception):
    """Raised when audio is invalid or cannot be processed due to user input"""
    pass

class TritonServerError(Exception):
    """Raised when Triton server is unavailable or times out"""
    pass

def get_part_embedding(audio_file, request_id="unknown", audio=Config.AUDIO):
    logger.info(f"[{request_id}] Starting embedding extraction for: {audio_file}")
    try:
        if not os.path.exists(audio_file):
            logger.error(f"[{request_id}] Audio file not found: {audio_file}")
            raise UserAudioError(f"Audio file not found: {audio_file}")
        
        # Round duration to 3 decimal places
        # duration = round(audio.get_duration(audio_file), 2)
        
        try:
            waveform, orig_sr = librosa.load(audio_file, sr=16000, mono=True)
        except Exception as e:
            logger.error(f"[{request_id}] Failed to load audio file: {e}")
            raise UserAudioError(f"Unable to read audio file - file may be corrupted or in an unsupported format")

        if len(waveform) == 0:
            logger.error(f"[{request_id}] Audio file is empty")
            raise UserAudioError("Audio file is empty or contains no data")
        
        if np.abs(waveform).max() < 1e-6:
            logger.error(f"[{request_id}] Audio is silent (max amplitude < 1e-6)")
            raise UserAudioError("Audio is silent or has no meaningful sound")     
        
        waveform = waveform / (np.abs(waveform).max() + 1e-8)
        # return embedding
        chunk_size = 16000
        embeddings_list = []

        logger.info(f"[{request_id}] Connecting to Triton server at {Config.MODEL_URL}")

        try:
            client = InferenceServerClient(
                url=Config.MODEL_URL,
                connection_timeout=Config.TRITON_TIMEOUT,
                network_timeout=Config.TRITON_TIMEOUT
            )
            if not client.is_server_live():
                logger.error(f"[{request_id}] Triton server is not live")
                raise TritonServerError("Inference server is not responding")
                
            if not client.is_server_ready():
                logger.error(f"[{request_id}] Triton server is not ready")
                raise TritonServerError("Inference server is not ready to process requests")
                
            if not client.is_model_ready(Config.MODEL_NAME):
                logger.error(f"[{request_id}] Model '{Config.MODEL_NAME}' is not ready")
                raise TritonServerError(f"Speaker recognition model is not available")

        except (ConnectionError, TimeoutError, InferenceServerException) as e:
            logger.error(f"[{request_id}] Failed to connect to Triton server: {e}")
            raise TritonServerError(f"Unable to connect to inference server")
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error creating Triton client: {e}")
            raise TritonServerError(f"Failed to initialize inference connection")

        num_chunks = (len(waveform) + chunk_size - 1) // chunk_size
        logger.info(f"[{request_id}] Processing {num_chunks} audio chunks")
        
        successful_chunks = 0
        failed_chunks = 0

        for i in range(0, len(waveform), chunk_size):
            chunk_idx = i // chunk_size
            chunk = waveform[i:i + chunk_size]
            
            # Skip chunks that are too short (less than 0.5 seconds)
            if len(chunk) < chunk_size // 2:
                continue
            
            # Pad the last chunk if it's shorter than chunk_size
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            # Reshape to (1, 16000) for batch dimension
            chunk = chunk.reshape(1, chunk_size).astype(np.float32)
            
            # Send to Triton
            try:
                inputs = [InferInput("audio_input", chunk.shape, "FP32")]
                inputs[0].set_data_from_numpy(chunk)
                outputs = [InferRequestedOutput("embeddings")]

                try:
                    response = client.infer(
                        model_name=Config.MODEL_NAME,
                        inputs=inputs,
                        outputs=outputs
                        # request_timeout=Config.REQUEST_TIMEOUT  # Triton request timeout
                    )
                    embedding = response.as_numpy("embeddings")
                    if embedding is not None and embedding.size > 0:
                        embeddings_list.append(embedding)
                        successful_chunks += 1
                    else:
                        failed_chunks += 1
                        logger.warning(f"[{request_id}] Chunk {chunk_idx} returned empty embedding")

                except InferenceServerException as e:
                    failed_chunks += 1
                    logger.error(f"[{request_id}] Triton inference error for chunk {e}")
                    continue
                except TimeoutError:
                    failed_chunks += 1
                   
                    # If too many timeouts, raise to prevent wasting time
                    if failed_chunks > num_chunks * 0.5:  # More than 50% failed
                        raise TritonServerError(f"Inference server is responding too slowly")

            except TritonServerError:
                raise  # Propagate Triton errors
            except Exception as e:
                failed_chunks += 1
                logger.error(f"[{request_id}] Unexpected error processing chunk {chunk_idx}: {e}")
                traceback.print_exc()
        
        
        if not embeddings_list:
            logger.error(f"[{request_id}] No valid embeddings extracted from audio")
            raise UserAudioError("Unable to extract voice features from audio - audio may be too short or contain only noise")
        
        # Average all embeddings to get a single representation
        final_embedding = np.mean(embeddings_list, axis=0)
        logger.info(f"[{request_id}] Final embedding shape: {final_embedding.shape}")
        
        return final_embedding

    except (UserAudioError, TritonServerError):
        raise  # Propagate specific errors
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in get_part_embedding: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Audio processing failed unexpectedly")

def recognition_audio_part(audio_path, request_id="unknown"):
    try:
        # Get embedding for the current audio segment
        print("starting embedding procedure ")
        embeddings = get_part_embedding(audio_file=audio_path, request_id=request_id)
        print("here embeddings done")
        
        embeddings = np.array(embeddings)  # make sure it's a numpy array
        if embeddings.ndim == 3:  # e.g., (1, 1, 192)
            embeddings = embeddings.squeeze(1)  # remove the middle dimension

        batch_embeddings = embeddings.astype(float).tolist()  # convert to list of 1D floats
        print("here embeddings done")


        # 2️⃣ Batch search (known → unknown → new unknown)
        batch_results = milvus_client.search_batch(
            batch_embeddings,
            similarity_threshold=Config.SIMILARITY_THRESHOLD,
        )

        if not batch_results:
            return "Unknown", 0.0

        result = batch_results[0]

        name = result["name"]
        score = result["similarity"]

        # 4️⃣ Store new unknown speaker only when required
        if result["is_unknown"] and result["should_store"]:
            milvus_client.add_unknown(
                unknown_id=name,
                embedding=batch_embeddings[0],
            )
            logger.info(f"[{request_id}] Stored new unknown speaker: {name}")

        logger.info(
            f"[{request_id}] SR result → name={name}, "
            f"score={score}, unknown={result['is_unknown']}"
        )

        return name, score

    except (UserAudioError, TritonServerError):
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error in recognition_audio_part: {e}")
        traceback.print_exc()
        raise RuntimeError("Speaker recognition process failed")
    

def is_audio_file(file_path: str) -> bool:
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",  # only audio streams
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    r = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = r.stdout.decode().strip()  # decode bytes to string
    return r.returncode == 0 and "audio" in output  # check if 'audio' appears anywhere


def extract_audio(video_path: str, audio_path: str):
    """
    Extract audio from video to WAV using ffmpeg.
    Tolerates minor stream corruption (common in downloaded MP4s).
    """
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",

        # ✅ Robustness flags:
        "-fflags", "+genpts+discardcorrupt",
        "-err_detect", "ignore_err",

        "-i", video_path,
        "-vn",
        "-map", "0:a:0?",          # ✅ if audio exists, take it; don't hard-fail mapping
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_path
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore")

        # ✅ If ffmpeg produced a partial wav, accept it if it's usable (> ~1 second)
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 16000 * 2:  # 1s mono s16
            return audio_path

        raise RuntimeError(f"Failed to extract audio from video: {stderr}")


def process_video(file, request_id="unknown"):
    # Create a unique temporary directory for this request
    temp_dir = tempfile.mkdtemp(prefix="speaker_recognition_")
    print("here process_video")
    try:
        # Read the uploaded video file into memory
        file_content = file.file.read()
        file_name, file_extension = os.path.splitext(file.filename)
        
        # Save the uploaded file in the temporary directory
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as temp_file:
            temp_file.write(file_content)

        # Check if the file has a video extension
        
        if any(file_extension.lower() == ext for ext in Config.VIDEO_EXTENSION):
            # Save the audio clip in the temporary directory
            audio_file_path = os.path.join(temp_dir, f"{file_name}.wav")
            
            try: 
                audio_file_path = extract_audio(file_path,audio_file_path)
            except RuntimeError as e:
                logger.error(f"[{request_id}] Video has no audio: {e}")
                raise UserAudioError("The uploaded video does not contain any audio track")
        else:
            if not is_audio_file(file_path):
                logger.error(f"[{request_id}] Invalid audio file format")
                raise UserAudioError("Invalid audio file format")
            audio_file_path = file_path
        person, score = recognition_audio_part(audio_file_path)
        # similarity = round((1-score)*100, 2)    
        # Process the audio file
        result = {"speaker_name": person}
        
        return result
    except (UserAudioError, TritonServerError):
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in process_video: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Video processing failed unexpectedly")
    
    finally:
        # Always clean up the temporary directory, regardless of success or failure
        shutil.rmtree(temp_dir, ignore_errors=True)
