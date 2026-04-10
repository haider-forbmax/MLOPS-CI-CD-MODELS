import os
## import shutil  # unused
## from scipy.spatial.distance import cdist  # unused
import numpy as np
## from pydub import AudioSegment, effects  # unused
## from pydub.silence import split_on_silence  # unused
from typing import List, Optional, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from config import Config
from pyannote.core import Segment
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
# import traceback
import time
import threading
## import uuid  # unused


def generate_speaker_id(name: str) -> str:
    timestamp = int(time.time() * 1000)
    return f"speaker_{name.lower().replace(' ', '_')}_{timestamp:x}"

class MilvusClient:
    def __init__(self):
        self.alias = Config.MILVUS_ALIAS
        self.known_collection: Optional[Collection] = None
        self.unknown_collection: Optional[Collection] = None

        # batching + counters
        self._pending_known = 0
        self._pending_unknown = 0
        self._last_flush_ts = time.time()

        self._flush_every_n = getattr(Config, "MILVUS_FLUSH_EVERY_N", 2000)
        self._flush_every_sec = getattr(Config, "MILVUS_FLUSH_EVERY_SECONDS", 60)

        self._counter_lock = threading.Lock()
        self._unknown_counter = 0

        self._connect()

    def _connect(self):
        connections.connect(
            alias=self.alias,
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT,
            db_name=Config.MILVUS_DB_NAME,
        )
        self._ensure_collections()
        self.known_collection.load()
        self.unknown_collection.load()
        self._unknown_counter = self._load_unknown_counter()
        
    def _ensure_collections(self):
        self._ensure_known()
        self._ensure_unknown()

    def _ensure_known(self):
        name = Config.MILVUS_COLLECTION_NAME

        if utility.has_collection(name, using=self.alias):
            self.known_collection = Collection(name, using=self.alias)
            return

        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("label_name", DataType.VARCHAR, max_length=128),
            FieldSchema("speaker_id", DataType.VARCHAR, max_length=128),
            FieldSchema("embeddings", DataType.FLOAT_VECTOR, dim=192),
        ]

        schema = CollectionSchema(fields, "Known speaker embeddings")
        self.known_collection = Collection(name, schema, using=self.alias)

        self.known_collection.create_index(
            "embeddings",
            {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )
        self.known_collection.create_index("id", {"index_type":"AUTOINDEX"})

    
    def _ensure_unknown(self):
        name = f"{Config.MILVUS_COLLECTION_NAME}_unknowns"

        if utility.has_collection(name, using=self.alias):
            self.unknown_collection = Collection(name, using=self.alias)
            return  

        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("unknown_id", DataType.VARCHAR, max_length=128),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=192),
        ]

        schema = CollectionSchema(fields, "Unknown speaker embeddings")
        self.unknown_collection = Collection(name, schema, using=self.alias)

        self.unknown_collection.create_index(
            "embedding",
            {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        self.unknown_collection.create_index("id", {"index_type":"AUTOINDEX"})
    
    def _load_unknown_counter(self) -> int:
        try:
            results = self.unknown_collection.query(
                expr="",
                limit=10000,
                output_fields=["unknown_id"],
            )
            mx = 0
            for r in results:
                uid = r.get("unknown_id", "")
                if uid.startswith("unknown_"):
                    mx = max(mx, int(uid.split("_")[1]))
            return mx
        except Exception:
            return 0

    def _next_unknown_id(self) -> str:
        with self._counter_lock:
            self._unknown_counter += 1
            return f"unknown_{self._unknown_counter}"

    def search_batch(
        self,
        embeddings: List[List[float]],
        similarity_threshold: float = Config.SIMILARITY_THRESHOLD,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not embeddings:
            return []

        # -------- search known -------- #
        known_hits = self.known_collection.search(
            data=embeddings,
            anns_field="embeddings",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["label_name"],
        )

        results: List[Optional[Dict[str, Any]]] = [None] * len(embeddings)
        need_unknown_idx = []

        for i, hits in enumerate(known_hits):
            for hit in hits:
                sim = float(hit.distance)
                if sim >= similarity_threshold:
                    results[i] = {
                        "name": hit.entity["label_name"],
                        "is_unknown": False,
                        "similarity": round(sim, 4),
                        "distance": round(1 - sim, 4),
                        "should_store": False,
                    }
                    break
            if results[i] is None:
                need_unknown_idx.append(i)

        # -------- search unknown -------- #
        if need_unknown_idx:
            unk_embeddings = [embeddings[i] for i in need_unknown_idx]
            unk_hits = self.unknown_collection.search(
                data=unk_embeddings,
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                output_fields=["unknown_id"],
            )

            unk_threshold = max(similarity_threshold - 0.05, 0.3)

            for idx, hits in zip(need_unknown_idx, unk_hits):
                for hit in hits:
                    sim = float(hit.distance)
                    if sim >= unk_threshold:
                        results[idx] = {
                            "name": hit.entity["unknown_id"],
                            "is_unknown": True,
                            "similarity": round(sim, 4),
                            "distance": round(1 - sim, 4),
                            "should_store": False,
                        }
                        break

        # -------- assign new unknowns -------- #
        for i in range(len(results)):
            if results[i] is None:
                uid = self._next_unknown_id()
                results[i] = {
                    "name": uid,
                    "is_unknown": True,
                    "similarity": 0.0,
                    "distance": 1.0,
                    "should_store": True,
                }

        return results

    def add_unknown(self, unknown_id: str, embedding: List[float]):
        self.unknown_collection.insert([[unknown_id], [embedding]])
        self._pending_unknown += 1
        self._maybe_flush()

    def add_speaker(self, label_name: str, embedding: List[float]):
        speaker_id = generate_speaker_id(label_name)
        self.known_collection.insert([[label_name], [speaker_id], [embedding]])
        self._pending_known += 1
        self._maybe_flush()

    def delete_speaker(self, name: str) -> int:
        expr = f'label_name == "{name}"'
        delete_result = self.known_collection.delete(expr)
        self.known_collection.flush()
        
        return getattr(delete_result, "delete_count", 0)

    def _maybe_flush(self, force: bool = False):
        now = time.time()
        if not force:
            if (self._pending_known + self._pending_unknown) < self._flush_every_n:
                if now - self._last_flush_ts < self._flush_every_sec:
                    return

        if self._pending_known:
            self.known_collection.flush()
            self._pending_known = 0

        if self._pending_unknown:
            self.unknown_collection.flush()
            self._pending_unknown = 0

        self._last_flush_ts = now

def training(label, audio_file):
    
    speaker_voice = Segment(0., float(Config.AUDIO.get_duration(audio_file)))
    waveform, sample_rate = Config.AUDIO.crop(audio_file, speaker_voice)

    waveform_segment = waveform_segment.astype(np.float32)
        
    client = InferenceServerClient(url=Config.MODEL_URL)
    inputs = [InferInput("audio_input", waveform_segment.shape, "FP32")]
    inputs[0].set_data_from_numpy(waveform_segment)
    outputs = [InferRequestedOutput("embeddings")]
    response = client.infer(Config.MODEL_NAME, inputs=inputs, outputs=outputs)
    embedding = response.as_numpy("embeddings")
    
    # embedding = model(waveform[None])
    
    # Create speaker data
    # speaker = [
    #     {"label_name": label, "embeddings": embedding}
    # ]
    print("embedding being saved")
    milvus_client.add_speaker(label, embedding.flatten().tolist())



milvus_client = MilvusClient()
