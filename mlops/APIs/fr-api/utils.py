import time
import uuid
import httpx
import base64
import tritonclient.http as httpclient
import io
from typing import List, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from config import Config
import threading
import numpy as np
import cv2

def generate_request_id() -> str:
    return f"pred_face_{uuid.uuid4().hex[:6]}"

def generate_face_id(name: str) -> str:
    timestamp = int(time.time() * 1000)
    return f"face_{name.lower().replace(' ', '_')}_{timestamp:x}"

def overlay_face_names(image: np.ndarray, detections: List) -> np.ndarray:
    """
    Draw bounding boxes and recognized names on image.
    Returns annotated OpenCV image.
    """

    # Convert OpenCV (BGR) to RGB for PIL
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)

    # Load font safely
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except:
        font = ImageFont.load_default()

    for det in detections:
        x1 = det.bounding_box.x1
        y1 = det.bounding_box.y1
        x2 = det.bounding_box.x2
        y2 = det.bounding_box.y2

        name = det.name if det.name != "unknown" else "Unknown"
        similarity = f"{det.similarity:.2f}"

        label = f"{name} ({similarity})"

        # Color logic
        if det.name != "unknown":
            box_color = (0, 200, 0)  # Green
        else:
            box_color = (200, 0, 0)  # Red

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        # Text position
        text_x = x1
        text_y = max(10, y1 - 30)

        text_bbox = draw.textbbox((text_x, text_y), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Draw background rectangle
        draw.rectangle(
            [text_x - 4, text_y - 4, text_x + text_w + 4, text_y + text_h + 4],
            fill=box_color
        )

        # Draw text
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

    # Convert back to OpenCV format
    annotated = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return annotated

class FaceRecognitionClient:
    def __init__(self):
        self.client = httpclient.InferenceServerClient(url=Config.FR_SERVICE_URL)
        self.model_name = Config.MODEL_NAME
        
    def extract(self, image: bytes) -> dict:
        """
        Returns:
            embeddings: np.array [total_faces, 512]d
            face_counts: list of face count per image
            confidences: list of confidence scores
            indices: list of image indices for each face
            bboxes: np.array [total_faces, 4] - [x1, y1, x2, y2]
            landmarks: np.array [total_faces, 5, 2] - 5 facial landmarks per face
                      Order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        # Prepare inputs as byte strings
        # Create input tensor
        inputs = [httpclient.InferInput("IMAGE_RAW", [1, 1], "BYTES")]
        
        # Pack bytes into numpy array
        np_bytes = np.array([[image]], dtype=object)
        inputs[0].set_data_from_numpy(np_bytes)
        
        # Request outputs
        outputs = [
            httpclient.InferRequestedOutput("EMBEDDINGS"),
            httpclient.InferRequestedOutput("FACE_COUNT"),
            httpclient.InferRequestedOutput("CONFIDENCES"),
            httpclient.InferRequestedOutput("FACE_INDICES"),
            httpclient.InferRequestedOutput("BBOXES"),
            httpclient.InferRequestedOutput("LANDMARKS")  # NEW
        ]
        
        results = self.client.infer(self.model_name, inputs, outputs=outputs)
        
        embeddings = results.as_numpy("EMBEDDINGS")
        norm_embeddings = []
        for embeds in embeddings:
            emb = np.array(embeds, dtype=np.float32)

            norm = np.linalg.norm(emb)
            # print(norm)
            if norm == 0:
                continue
            emb = emb / norm
            norm_embeddings.append(emb)
        return {
            "embeddings": norm_embeddings,
            "face_counts": results.as_numpy("FACE_COUNT").flatten().tolist(),
            "confidences": results.as_numpy("CONFIDENCES").flatten().tolist(),
            "indices": results.as_numpy("FACE_INDICES").flatten().tolist(),
            "bboxes": results.as_numpy("BBOXES"),
            "landmarks": results.as_numpy("LANDMARKS")  # [N, 5, 2]
        }

class MilvusClient:

    def __init__(self):
        self._alias = Config.MILVUS_ALIAS
        self.known_collection: Optional[Collection] = None
        self.unknown_collection: Optional[Collection] = None
        # Batched flush controls
        self._pending_unknown_writes = 0
        self._pending_known_writes = 0
        self._last_flush_ts = 0.0
        
        self._counter_lock = threading.Lock()

        # Tune these
        self._flush_every_n_writes = getattr(Config, "MILVUS_FLUSH_EVERY_N_WRITES", 2000)
        self._flush_every_seconds = getattr(Config, "MILVUS_FLUSH_EVERY_SECONDS", 60.0)

        self._connect()
        self._unknown_counter = self._load_unknown_counter()

    def _connect(self):
        connections.connect(
            alias=self._alias,
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT,
            db_name=Config.DATABASE_NAME,
        )
        self._ensure_collections()

    def _ensure_collections(self):
        self._ensure_known_collection()
        self._ensure_unknown_collection()

        # Load once (do NOT load/unload per request)
        self.known_collection.load()
        self.unknown_collection.load()
    
    def _ensure_known_collection(self):
        """Setup known faces collection"""
        collection_name = Config.MILVUS_COLLECTION_NAME  # e.g. "face_embeddings"

        if utility.has_collection(collection_name, using=self._alias):
            self.known_collection = Collection(collection_name, using=self._alias)
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="face_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="face_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=Config.MILVUS_VECTOR_DIM),
        ]
        schema = CollectionSchema(fields, "Face recognition embeddings")
        self.known_collection = Collection(collection_name, schema, using=self._alias) 
            # Create index for fast similarity search
        id_index_params = {"index_type":"AUTOINDEX"}
        self.known_collection.create_index("id", id_index_params)
        index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        self.known_collection.create_index("embedding", index_params)

    def _ensure_unknown_collection(self):
        collection_name = f"{Config.MILVUS_COLLECTION_NAME}_unknowns"

        if utility.has_collection(collection_name, using=self._alias):
            self.unknown_collection = Collection(collection_name, using=self._alias)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                # Stable daily ID: "unknown_1", "unknown_2", ...
                FieldSchema(name="unknown_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=Config.MILVUS_VECTOR_DIM),
            ]
            schema = CollectionSchema(fields, "Unknown face embeddings ")
            self.unknown_collection = Collection(collection_name, schema, using=self._alias)

            # HNSW is typically faster for small/medium "frequently searched" sets like daily unknowns.
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }
            id_index_params = {"index_type":"AUTOINDEX"}
            self.unknown_collection.create_index("id", id_index_params)
            self.unknown_collection.create_index("embedding", index_params)

        # Ensure today's partition exists

    def _maybe_flush(self, force: bool = False) -> None:
        now = time.time()
        if not force:
            if (self._pending_unknown_writes + self._pending_known_writes) < self._flush_every_n_writes:
                if (now - self._last_flush_ts) < self._flush_every_seconds:
                    return

        # Flush both if needed (flush is expensive; call sparingly)
        try:
            if self._pending_known_writes:
                self.known_collection.flush()
                self._pending_known_writes = 0
            if self._pending_unknown_writes:
                self.unknown_collection.flush()
                self._pending_unknown_writes = 0
        finally:
            self._last_flush_ts = now

    def _load_unknown_counter(self) -> int:
        """
        On startup, continue unknown numbering from existing records.
        """
        try:
            results = self.unknown_collection.query(
                expr="",
                limit=10000,
                output_fields=["unknown_id"],
            )
            max_num = 0
            for r in results:
                uid = r.get("unknown_id", "")
                if uid.startswith("unknown_"):
                    try:
                        n = int(uid.split("_", 1)[1])
                        max_num = max(max_num, n)
                    except ValueError:
                        pass
            return max_num
        except Exception:
            return 0

    def _get_next_unknown_id_fast(self) -> str:
        with self._counter_lock:
            self._unknown_counter += 1
            return f"unknown_{self._unknown_counter}"

    def flush_now(self) -> None:
        """Call this once per request/frame (optional) if you want stronger read-after-write behavior."""
        self._maybe_flush(force=True)
    
    def _search_known(self, embeddings: List[List[float]], top_k: int, similarity_threshold: float) -> List[Optional[Dict[str, Any]]]:
        """
        Returns best known match per embedding if above threshold, else None.
        """
        if not embeddings:
            return []

        results = self.known_collection.search(
            data=embeddings,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["face_name", "face_id"],
        )

        out: List[Optional[Dict[str, Any]]] = []
        for hits in results:
            best = None
            for hit in hits:
                sim = float(hit.distance)  # cosine similarity (higher is better) in your usage
                # print(f'name: {hit.entity.get("face_name")}  sim: {sim}')
                if sim >= similarity_threshold:
                    best = {
                        "matched": True,
                        "name": hit.entity.get("face_name"),
                        "face_id": hit.entity.get("face_id"),
                        "similarity": round(sim, 4),
                        "distance": round(1 - sim, 4),
                        "is_unknown": False,
                        "should_store": False,
                    }
                    break
            out.append(best)
        return out
        
    def _search_unknown(
        self,
        embeddings: List[List[float]],
        top_k: int,
        similarity_threshold: float
    ) -> List[Optional[Dict[str, Any]]]:

        if not embeddings:
            return []

        unknown_threshold = max(similarity_threshold - 0.05, 0.3)

        results = self.unknown_collection.search(
            data=embeddings,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["unknown_id"],
        )

        out = []
        for hits in results:
            best = None
            for hit in hits:
                sim = float(hit.distance)
                if sim >= unknown_threshold:
                    best = {
                        "matched": True,
                        "name": hit.entity.get("unknown_id"),
                        "face_id": hit.entity.get("unknown_id"),
                        "similarity": round(sim, 4),
                        "distance": round(1 - sim, 4),
                        "is_unknown": True,
                        "should_store": False,  # no overwrite needed
                    }
                    break
            out.append(best)

        return out

    def search(self, embedding: List[float], similarity_threshold: float = Config.SIMILARITY_THRESHOLD, top_k: int = 5) -> List[Dict[str, Any]]:
        """Single-embedding search (kept for compatibility)."""
        return self.search_batch([embedding], similarity_threshold=similarity_threshold, top_k=top_k)[0:1][0] 
    
    def search_batch(self, embeddings: List[List[float]], similarity_threshold: float = Config.SIMILARITY_THRESHOLD, top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Batch search:
          - search known for all embeddings (1 call)
          - for those not matched, search unknowns (1 call)
          - assign new unknown IDs in-memory (O(1)) for remaining
        Returns: list of matches list per embedding (same shape as your old code).
        """
        if not embeddings:
            return []

        # Stage 1: knowns
        known_best = self._search_known(embeddings, top_k=top_k, similarity_threshold=similarity_threshold)

        # Prepare unknown search for those without known match
        idx_need_unknown = [i for i, k in enumerate(known_best) if k is None]
        unknown_best_map: Dict[int, Optional[Dict[str, Any]]] = {}
        
        if idx_need_unknown:
            emb_need_unknown = [embeddings[i] for i in idx_need_unknown]
            unknown_best = self._search_unknown(emb_need_unknown, top_k=top_k, similarity_threshold=similarity_threshold)
            for i, best in zip(idx_need_unknown, unknown_best):
                unknown_best_map[i] = best

        # Build final matches
        final: List[List[Dict[str, Any]]] = []
        for i in range(len(embeddings)):
            if known_best[i] is not None:
                final.append([known_best[i]])
                continue

            ubest = unknown_best_map.get(i)
            if ubest is not None:
                final.append([ubest])
                continue

            # Brand new unknown: O(1) counter
            new_unknown_id = self._get_next_unknown_id_fast()
            final.append([{
                "matched": False,
                "name": new_unknown_id,
                "face_id": new_unknown_id,
                "similarity": 0.0,
                "distance": 1.0,
                "is_unknown": True,
                "should_store": True,
            }])

        return final

    def add_unknown(self, unknown_id: str, embedding: List[float]) -> None:
        # self.unknown_collection.insert([
        #     [unknown_id],
        #     [embedding],
        # ])
        # print(f'unknown_id: {unknown_id} ')
        self.unknown_collection.insert([{
            # "id": None,
            "unknown_id": unknown_id,
            "embedding": embedding,
        }])
        self._pending_unknown_writes += 1
        self._maybe_flush(force=False)
    
    def add_face(self, name: str, face_id: str, embedding: List[float]) -> str:
        data = [[name], [face_id], [embedding]]
        insert_result = self.known_collection.insert(data)
        self.known_collection.flush()
        
        return insert_result.primary_keys[0]

    def delete_face(self, name: str) -> int:
        expr = f'face_name == "{name}"'
        delete_result = self.known_collection.delete(expr)
        self.known_collection.flush()
        
        return getattr(delete_result, "delete_count", 0)

    def get_all_trained_labels(self):
        if not self.known_collection:
            return {}
        results = self.known_collection.query(expr="",limit=10000, output_fields = ["face_name"],)

        labels: set[str] = set()
        for r in results:
            name = r.get("face_name")

            if not name:
                continue

            labels.add(name)

        return sorted(labels)


    def get_face_ids_by_name(self, name: str) -> List[str]:
        expr = f'face_name == "{name}"'
        results = self.known_collection.query(expr, output_fields=["face_id"])
        return [r["face_id"] for r in results]

face_client = FaceRecognitionClient()
milvus_client = MilvusClient()
