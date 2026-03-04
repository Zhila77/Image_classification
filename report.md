# EAIGLE AI — Multi-Camera Computer Vision Pipeline
## Technical Report

**Project:** Real-Time Multi-Camera CV Pipeline
**Date:** 2026-03-04
**Stack:** Python 3.12, asyncio, Redis Streams, YOLOv8, FastAPI, Docker

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Component-Level Breakdown](#3-component-level-breakdown)
4. [Data Flow Description](#4-data-flow-description)
5. [Deployment Strategy](#5-deployment-strategy)
6. [Configuration Reference](#6-configuration-reference)
7. [How to Run](#7-how-to-run)

---

## 1. Project Overview

EAIGLE AI is a scalable real-time computer vision pipeline designed to ingest video streams from up to **50 simultaneous RTSP cameras**, run object detection and OCR, and produce structured event hypotheses for downstream consumers (dashboards, alert systems, audit logs).

### Key Requirements Addressed

| Requirement | Solution |
|---|---|
| 50 simultaneous RTSP cameras | asyncio + ThreadPoolExecutor per camera |
| Real-time preprocessing | ProcessPoolExecutor (GIL bypass, 4–8 CPU cores) |
| Zero-copy frame transfer | POSIX shared memory (`/dev/shm`) |
| At-least-once delivery | Redis Streams with consumer groups + XACK |
| Object detection | YOLOv8 (configurable: nano → xlarge) |
| Multi-stage cascade | Stage A (full frame) → Stage B (crops: OCR/classify) |
| Hypothesis deduplication | SpatialNMS + ConfidenceVoting + TemporalSmoother |
| GPU hardware decode (optional) | NVIDIA DeepStream backend |
| Observability | Prometheus metrics + structured JSON logs |
| Containerized deployment | Docker Compose (dev) / Kubernetes (prod) |

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EAIGLE AI PIPELINE                                   │
│                                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         ┌──────────┐            │
│  │  CAM_01  │  │  CAM_02  │  │  CAM_03  │   ...   │  CAM_50  │            │
│  │  RTSP    │  │  RTSP    │  │  RTSP    │         │  RTSP    │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘         └────┬─────┘            │
│       └──────────────┴──────────────┴────────────────────┘                 │
│                              │                                              │
│                    ┌─────────▼──────────┐                                  │
│                    │   INGESTION LAYER   │                                  │
│                    │  ┌───────────────┐ │                                  │
│                    │  │OpenCV+FFmpeg  │ │  ← asyncio + ThreadPoolExecutor  │
│                    │  │  OR DeepStream│ │  ← NVDEC hardware decode (GPU)   │
│                    │  └───────────────┘ │                                  │
│                    └─────────┬──────────┘                                  │
│                              │  frame pixels (6 MB per frame)              │
│              ┌───────────────┼────────────────┐                            │
│              ▼               ▼                ▼                            │
│       /dev/shm          /dev/shm          /dev/shm   ← POSIX Shared Mem   │
│    eaigle_frame_1    eaigle_frame_2    eaigle_frame_3                      │
│              └───────────────┴────────────────┘                            │
│                              │  metadata pointer only (~200 bytes)         │
│                    ┌─────────▼──────────┐                                  │
│                    │   Redis Streams     │  raw_frames:{cam_id}            │
│                    └─────────┬──────────┘                                  │
│                              │                                              │
│                    ┌─────────▼──────────┐                                  │
│                    │ PREPROCESSING LAYER │                                  │
│                    │  ProcessPoolExecutor│  ← 4–8 workers (GIL bypass)     │
│                    │  color_convert      │                                  │
│                    │  resize → 640×640   │                                  │
│                    │  normalize float32  │                                  │
│                    └─────────┬──────────┘                                  │
│                              │  metadata pointer (new shm block)           │
│                    ┌─────────▼──────────┐                                  │
│                    │   Redis Streams     │  preprocessed_frames             │
│                    └─────────┬──────────┘                                  │
│                              │                                              │
│                    ┌─────────▼──────────┐                                  │
│                    │ INFERENCE DISPATCH  │                                  │
│                    │  DynamicBatcher     │  ← batch ≤8 OR ≤50ms timeout    │
│                    └─────────┬──────────┘                                  │
│                              │  HTTP batch request (base64 frames)         │
│              ┌───────────────┴────────────────┐                            │
│              ▼                                ▼                            │
│   ┌──────────────────┐             ┌──────────────────┐                   │
│   │ /detect/primary  │             │ /detect/secondary│                   │
│   │  YOLOv8 Object   │──crops──►   │  OCR / Classify  │                   │
│   │  Detection       │             │  (license plates,│                   │
│   │  (vehicle,person)│             │   badges, etc.)  │                   │
│   └────────┬─────────┘             └────────┬─────────┘                   │
│            │    FastAPI Microservice (port 8001)    │                      │
│            └───────────────┬────────────────┘                             │
│                            │  StageResult JSON                             │
│                  ┌─────────▼──────────┐                                   │
│                  │   Redis Streams     │  inference_results                │
│                  └─────────┬──────────┘                                   │
│                            │                                               │
│                  ┌─────────▼──────────┐                                   │
│                  │ AGGREGATION LAYER   │                                   │
│                  │  SpatialNMS         │  ← IoU > 0.45 suppression        │
│                  │  ConfidenceVoting   │  ← score < 0.40 filter           │
│                  │  TemporalSmoother   │  ← 5-frame sliding window        │
│                  └─────────┬──────────┘                                   │
│                            │  Hypothesis events                            │
│                  ┌─────────▼──────────┐                                   │
│                  │   Redis PubSub      │  hypotheses channel               │
│                  └─────────┬──────────┘                                   │
│           ┌────────────────┼──────────────────┐                           │
│           ▼                ▼                  ▼                           │
│     Dashboard          Alert System       Audit Log                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component-Level Breakdown

### 3.1 Ingestion Layer

| Component | File | Role |
|---|---|---|
| `CameraWorker` | `src/eaigle/ingestion/camera_worker.py` | One asyncio coroutine per camera. Reads RTSP via `cv2.VideoCapture` offloaded to `ThreadPoolExecutor`. Auto-reconnects on failure. |
| `CameraStreamManager` | `src/eaigle/ingestion/camera_manager.py` | Supervises all N camera coroutines. Restarts crashed tasks every 2s. |
| `DeepStreamIngestionManager` | `src/eaigle/ingestion/deepstream_manager.py` | Optional NVIDIA DeepStream backend. `nvurisrcbin → nvstreammux → RGBA pad probe`. Hardware NVDEC decode on GPU. |

**Key design decision:** OpenCV's `cap.read()` is a blocking C call that cannot be awaited. Solution: `loop.run_in_executor(thread_pool, cap.read)` releases the event loop during the blocking call, allowing other coroutines to run concurrently. This enables 50 cameras to run in a single Python process.

**DeepStream backend** (activated by setting `ingestion.backend: deepstream` in config):
- GStreamer pipeline runs in a dedicated GLib thread
- `nvurisrcbin` handles RTSP decode using NVDEC (hardware, zero CPU)
- `nvstreammux` batches multiple streams on the GPU
- Pad probe on `nvstreammux.src` extracts RGBA frames → numpy RGB
- `asyncio.run_coroutine_threadsafe()` bridges GLib thread → asyncio event loop
- Falls back to OpenCV automatically if `gi`/`pyds` not installed

---

### 3.2 Shared Memory Transport

| Component | File | Role |
|---|---|---|
| `write_frame_to_shm` | `src/eaigle/preprocessing/shm_utils.py` | Allocates `/dev/shm/eaigle_{frame_id}`, copies numpy frame bytes. Returns shm name (the "pointer"). |
| `read_frame_from_shm` | `src/eaigle/preprocessing/shm_utils.py` | Maps block, copies frame out, unlinks (frees) the block. |
| `cleanup_stale_shm` | `src/eaigle/preprocessing/shm_utils.py` | Background loop — unlinks blocks older than 10s to prevent memory leaks. |

**Why shared memory instead of Redis for frame data?**

A 1920×1080 BGR frame is ~6 MB. Without shared memory:
- 50 cameras × 10 FPS × 6 MB = **3 GB/s through Redis** → impossible
- Redis would become a bottleneck and RAM would be exhausted

With shared memory:
- Frame pixels stay in `/dev/shm` (kernel RAM, fastest possible access)
- Redis carries only a 200-byte metadata message (the shm name + shape + dtype)
- Preprocessing reads the frame with a memory map — **zero copy**
- After reading, the block is unlinked (freed) immediately

---

### 3.3 Preprocessing Layer

| Component | File | Role |
|---|---|---|
| `PreprocessingWorkerPool` | `src/eaigle/preprocessing/worker_pool.py` | `ProcessPoolExecutor` with 4–8 workers. Each worker process holds a `PreprocessingPipeline` instance. Bounds in-flight tasks to `num_workers × 4` for backpressure. |
| `PreprocessingPipeline` | `src/eaigle/preprocessing/pipeline.py` | Composable chain of ops built from YAML config. Stateless, safe across processes. |
| `ColorConvertOp` | `src/eaigle/preprocessing/ops/color_convert.py` | `cv2.cvtColor(BGR → RGB)` |
| `ResizeOp` | `src/eaigle/preprocessing/ops/resize.py` | `cv2.resize` to 640×640 (YOLO input size) |
| `NormalizeOp` | `src/eaigle/preprocessing/ops/normalize.py` | Divide by 255, convert to float32 |
| `GaussianDenoiseOp` | `src/eaigle/preprocessing/ops/noise_reduction.py` | `cv2.GaussianBlur` |
| `FastNLMeansDenoiseOp` | `src/eaigle/preprocessing/ops/noise_reduction.py` | `cv2.fastNlMeansDenoisingColored` (higher quality, slower) |

**Why ProcessPoolExecutor?**
Python's GIL prevents true CPU parallelism with threads. `ProcessPoolExecutor` spawns separate OS processes, each with their own GIL, enabling genuine parallel preprocessing across CPU cores. A `_pool_init` initializer builds the pipeline once per worker process — not once per frame.

---

### 3.4 Inference Layer

| Component | File | Role |
|---|---|---|
| `DynamicBatcher` | `src/eaigle/inference/batcher.py` | Dual-trigger batch assembly. Fires on: size ≥ `max_size` OR `max_wait_ms` timeout. |
| `InferenceDispatcher` | `src/eaigle/inference/dispatcher.py` | Reads `preprocessed_frames`, feeds batcher. On flush: reads pixels from shm, base64-encodes, POSTs to inference service. Extracts crops for Stage B. |
| Inference Service | `inference_service/app.py` | FastAPI microservice. Stage A: YOLOv8 object detection. Stage B: OCR/classification stub. |

**DynamicBatcher dual-trigger logic:**

```
Frame arrives → add to batch
├── if len(batch) >= max_size (8):   → flush immediately
└── if batch just became non-empty:  → start 50ms countdown timer
    └── when timer expires:          → flush whatever is in batch
```

This ensures:
- Under high load (50 cameras): batches fill quickly → GPU throughput maximized
- Under low load (few cameras): no frame waits more than 50ms

**Without batching:** 50 cameras × 10 FPS = 500 HTTP requests/sec
**With batch size 8:** ≤ 63 HTTP requests/sec, each processing 8 frames in parallel on GPU

---

### 3.5 YOLOv8 Object Detection (Stage A)

Model: **YOLOv8n** (nano) by default — configurable via `YOLO_MODEL` environment variable.

| Model | File Size | Approx Speed (CPU) | mAP50 |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | ~50ms/frame | 37.3 |
| `yolov8s.pt` | 22 MB | ~90ms/frame | 44.9 |
| `yolov8m.pt` | 52 MB | ~200ms/frame | 50.2 |
| `yolov8l.pt` | 87 MB | ~400ms/frame | 52.9 |
| `yolov8x.pt` | 130 MB | ~700ms/frame | 53.9 |

**COCO → Pipeline label mapping:**

| YOLO COCO Class | Pipeline Label |
|---|---|
| car, bus | vehicle |
| truck | truck |
| person | person |
| bicycle | bicycle |
| motorcycle | motorcycle |
| traffic light | traffic_light |
| stop sign | stop_sign |

All other COCO classes (airplane, boat, dog, etc.) are filtered out as irrelevant to the security/surveillance domain.

---

### 3.6 Aggregation & Fusion Layer

| Component | File | Strategy |
|---|---|---|
| `ResultCollector` | `src/eaigle/aggregation/result_collector.py` | Buffers Stage A + Stage B results per `frame_id`. Marks frame complete when both stages are received. Flushes incomplete frames after 2s timeout. |
| `SpatialNMS` | `src/eaigle/aggregation/fusion_strategies/spatial_nms.py` | Non-Maximum Suppression — removes duplicate bounding boxes with IoU > 0.45. Keeps highest confidence detection. |
| `ConfidenceVoting` | `src/eaigle/aggregation/fusion_strategies/confidence_voting.py` | Filters detections below confidence threshold (0.40). |
| `TemporalSmoother` | `src/eaigle/aggregation/fusion_strategies/temporal_smoother.py` | Per-camera per-label `deque(maxlen=5)`. Label confirmed only if it appears in ≥2 of last 5 frames. Eliminates single-frame false positives. |
| `HypothesisFusion` | `src/eaigle/aggregation/hypothesis_fusion.py` | Orchestrates NMS → voting → smoothing. Produces human-readable `event_description`. |
| `HypothesisStore` | `src/eaigle/aggregation/hypothesis_store.py` | Persists hypothesis to Redis `HSET hypothesis:{id}` (60s TTL) + `LPUSH camera_hypotheses:{cam_id}` (keep 100) + `PUBLISH hypotheses`. |

**Fusion pipeline order:**
```
Raw detections (from Stage A + B)
    │
    ▼
SpatialNMS        → removes: same object detected twice with overlapping boxes
    │
    ▼
ConfidenceVoting  → removes: low-confidence / uncertain detections
    │
    ▼
TemporalSmoother  → removes: single-frame flickers / false positives
    │
    ▼
Confirmed Hypothesis → "Detected at Gate 1 [cam_01]: bus (vehicle, 91%), 3× person"
```

---

### 3.7 Data Models

| Model | File | Fields |
|---|---|---|
| `FrameMetadata` | `src/eaigle/models/frame.py` | `frame_id, camera_id, capture_ts, sequence_num, shm_key, width, height, channels, dtype, pipeline_stage` |
| `BoundingBox` | `src/eaigle/models/detection.py` | `x1, y1, x2, y2` (normalized 0–1). Methods: `.iou()`, `.to_pixel_coords(w,h)` |
| `Detection` | `src/eaigle/models/detection.py` | `stage, label, confidence, bbox, ocr_text, parent_detection_id` |
| `Hypothesis` | `src/eaigle/models/hypothesis.py` | `frame_id, camera_id, detections[], event_description, confidence, pipeline_latency_ms` |
| `CameraConfig` | `src/eaigle/models/hypothesis.py` | `camera_id, rtsp_url, target_fps, width, height, zone` |

---

### 3.8 Transport Layer

| Component | File | Role |
|---|---|---|
| `RedisClient` | `src/eaigle/transport/redis_client.py` | Shared async connection pool. `decode_responses=True`. |
| `StreamProducer` | `src/eaigle/transport/stream_producer.py` | `XADD` with approximate trimming at 500 messages. Idempotent `ensure_group()`. |
| `StreamConsumer` | `src/eaigle/transport/stream_consumer.py` | `XREADGROUP` async iterator. `ack()` after successful processing. |

**Redis Stream names:**

| Stream | Producer | Consumer | Content |
|---|---|---|---|
| `raw_frames:{camera_id}` | CameraWorker | PreprocessingWorkerPool | Raw frame metadata pointer |
| `preprocessed_frames` | PreprocessingWorkerPool | InferenceDispatcher | Preprocessed frame metadata pointer |
| `inference_results` | InferenceDispatcher | ResultCollector | StageResult JSON |
| `hypotheses` (PubSub) | HypothesisStore | Dashboard / Alerts | Final hypothesis events |

---

### 3.9 Observability

| Metric | Type | Description |
|---|---|---|
| `eaigle_frames_ingested_total` | Counter | Frames captured per camera |
| `eaigle_frames_dropped_total` | Counter | Frames dropped (backpressure) |
| `eaigle_preprocessing_latency_ms` | Histogram | Time spent in preprocessing per frame |
| `eaigle_inference_latency_ms` | Histogram | Round-trip time to inference service |
| `eaigle_batch_size` | Histogram | Batch sizes sent to inference |
| `eaigle_detections_total` | Counter | Detections by label |
| `eaigle_hypotheses_total` | Counter | Hypotheses published |
| `eaigle_active_cameras` | Gauge | Currently connected cameras |
| `eaigle_shm_blocks_active` | Gauge | Live shared memory blocks |
| `eaigle_pipeline_latency_ms` | Histogram | End-to-end: capture → hypothesis |
| `eaigle_redis_publish_errors_total` | Counter | Redis write failures |

Prometheus scrapes port **9090**. Grafana can be pointed at this for dashboards.

---

## 4. Data Flow Description

### 4.1 Frame Lifecycle (Single Frame, End-to-End)

```
T=0ms    Camera captures frame
          1920×1080 BGR, uint8, ~6 MB
          │
T=1ms    CameraWorker.run()
          → write_frame_to_shm(frame_id, frame)
          → Creates /dev/shm/eaigle_<uuid> (6MB block)
          → Publishes FrameMetadata to Redis:
            XADD raw_frames:cam_01 * {
              frame_id: "abc-123",
              shm_key:  "eaigle_abc-123",
              width: 1920, height: 1080, channels: 3,
              dtype: "uint8",
              capture_ts: 1709...,
              pipeline_stage: "raw"
            }
          │
T=2ms    PreprocessingWorkerPool
          → XREADGROUP raw_frames:cam_01 (consumer group)
          → Worker process picks up message
          → read_frame_from_shm("eaigle_abc-123", (1080,1920,3), uint8)
            [maps 6MB block, copies out, unlinks — old block freed]
          → Pipeline: BGR→RGB → resize(640,640) → normalize(÷255 → float32)
          → write_frame_to_shm("pp_abc-123", processed_frame)
            [new /dev/shm/eaigle_pp_abc-123 block, 640×640×3×4 = 4.9MB]
          → XACK raw_frames:cam_01 <msg_id>
          → XADD preprocessed_frames * {
              shm_key: "eaigle_pp_abc-123",
              width: 640, height: 640,
              dtype: "float32", ...
            }
          │
T=5ms    InferenceDispatcher
          → XREADGROUP preprocessed_frames
          → Frame enters DynamicBatcher queue
          │
          (waits for batch: size=8 OR 50ms timeout)
          │
T=55ms   DynamicBatcher flushes batch of N frames
          → For each frame: read_frame_from_shm → base64 encode
          → HTTP POST http://inference:8001/detect/primary
            body: {batch_id: "...", frames: [{frame_id, data_b64, shape, dtype}, ...]}
          │
T=75ms   YOLOv8 runs on batch (GPU or CPU)
          → Returns:
            [{label:"bus",    conf:0.91, bbox:{x1:0.1,y1:0.2,x2:0.8,y2:0.9}},
             {label:"person", conf:0.87, bbox:{...}},
             {label:"person", conf:0.79, bbox:{...}}]
          │
T=76ms   Dispatcher extracts crops for Stage B
          → For each "vehicle"/"person" detection:
            crop = frame[y1:y2, x1:x2]
          → HTTP POST /detect/secondary
            body: {frames: [{data_b64: crop_b64, label_hint: "vehicle", ...}]}
          → Secondary returns: {label:"license_plate", ocr_text:"ABC123", conf:0.88}
          │
T=80ms   Dispatcher publishes StageResults to Redis:
          XADD inference_results * {
            capture_ts: "1709...",
            result_json: "{stage: primary, detections: [...]}"
          }
          XADD inference_results * {
            capture_ts: "1709...",
            result_json: "{stage: secondary, detections: [...]}"
          }
          │
T=81ms   ResultCollector receives Stage A + Stage B
          → FrameAccumulator groups by frame_id
          → When both stages present → triggers HypothesisFusion
          │
T=82ms   HypothesisFusion
          1. SpatialNMS:        [bus(0.91), person(0.87), person(0.79)] → same (no overlap)
          2. ConfidenceVoting:  all > 0.40 → all pass
          3. TemporalSmoother:  "bus" seen in 3 of last 5 frames → confirmed
          → event_description: "Detected at Gate 1 [cam_01]: bus (vehicle, 91%), 2× person"
          │
T=83ms   HypothesisStore
          → HSET hypothesis:xyz {frame_id, camera_id, detections, event_description, ...}
             (TTL: 60 seconds)
          → LPUSH camera_hypotheses:cam_01 "xyz"  (keep latest 100)
          → PUBLISH hypotheses {hypothesis JSON}
          │
          ✓ End-to-end latency: ~83ms
          ✓ Event available to all downstream subscribers
```

### 4.2 Concurrency Model

```
Main Process (asyncio event loop)
├── CameraWorker × 50          (coroutines, I/O-bound)
│   └── ThreadPoolExecutor     (for blocking cv2.read calls)
│
├── PreprocessingWorkerPool    (asyncio consumer)
│   └── ProcessPoolExecutor    (4–8 processes, CPU-bound)
│       ├── Worker 0: pipeline init + frame processing
│       ├── Worker 1: pipeline init + frame processing
│       └── ...
│
├── InferenceDispatcher        (coroutine, I/O-bound)
│   └── DynamicBatcher         (size OR time trigger)
│       └── httpx.AsyncClient  (HTTP to inference service)
│
├── ResultCollector            (coroutine)
│   └── HypothesisFusion       (in-process, fast)
│       └── HypothesisStore    (Redis writes)
│
└── Background tasks
    ├── SHM cleanup loop       (every 10s)
    └── Prometheus metrics     (port 9090)
```

### 4.3 Backpressure Mechanism

If preprocessing falls behind ingestion:
1. `PreprocessingWorkerPool` bounds in-flight tasks to `num_workers × 4`
2. When limit reached → camera workers check `BACKPRESSURE_THRESHOLD` (300 messages in Redis stream)
3. If stream is full → `CameraWorker` skips publishing (drops frame rather than OOMing)
4. `eaigle_frames_dropped_total` Prometheus counter increments

---

## 5. Deployment Strategy

### 5.1 Local Development

Prerequisites: Docker (for Redis), Python 3.11+

```bash
# Clone and install
cd EAIGLE_AI
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install ultralytics matplotlib

# Terminal 1 — Redis
docker run -d --name eaigle-redis -p 6379:6379 redis:7-alpine \
  --maxmemory 256mb --maxmemory-policy allkeys-lru

# Terminal 2 — Inference service (downloads YOLOv8n on first run ~6MB)
uvicorn inference_service.app:app --host 0.0.0.0 --port 8001

# Terminal 3 — Camera simulator (synthetic frames with real image if available)
#   Place any JPEG at /tmp/sample_frame.jpg for real YOLO detections
python scripts/simulate_cameras.py --cameras 5 --fps 10

# Terminal 4 — Main pipeline
python -m eaigle.app --config configs/pipeline.yaml

# Run tests
pytest tests/unit/ -v
```

**Expected output (with sample image):**
```
INFO  eaigle.aggregation.hypothesis_store [cam_01] Detected at Gate 1 (cam_01):
      bus (vehicle, 91%), person (person, 87%), person (person, 79%)
      (conf=0.86, latency=83ms)
```

---

### 5.2 Docker Compose (Staging / Demo)

```bash
cd docker
docker compose up
```

**Services:**

| Service | Image | Port | Notes |
|---|---|---|---|
| `redis` | `redis:7-alpine` | 6379 | 256MB maxmemory, LRU eviction |
| `inference` | `Dockerfile.inference` | 8001 | YOLOv8n, 2 uvicorn workers |
| `pipeline` | `Dockerfile.pipeline` | — | Main asyncio app |
| `simulator` | same as pipeline | — | 5 synthetic cameras @ 10 FPS |
| `prometheus` | `prom/prometheus:latest` | 9090 | Scrapes pipeline metrics |

**DeepStream variant** (requires NVIDIA GPU + drivers):
```bash
docker compose --profile deepstream up
```

**Useful commands:**
```bash
# View logs
docker compose logs -f pipeline

# Check Redis stream depth
docker exec eaigle-redis redis-cli XLEN preprocessed_frames

# Stop everything
docker compose down -v
```

---

### 5.3 Kubernetes (Production)

**Recommended cluster topology for 50 cameras:**

```
Namespace: eaigle
│
├── redis-cluster (StatefulSet)
│   ├── 3 replicas with Redis Sentinel for HA
│   └── PersistentVolumeClaim: 10Gi SSD
│
├── inference-service (Deployment)
│   ├── replicas: 2–8 (auto-scaled)
│   ├── nodeSelector: cloud.google.com/gke-accelerator: nvidia-tesla-t4
│   ├── resources.limits: {nvidia.com/gpu: 1, memory: 8Gi}
│   └── HPA: scale on GPU utilization > 70%
│
├── pipeline (Deployment)
│   ├── replicas: 5 (10 cameras per pod)
│   ├── env: CAMERAS_PER_INSTANCE=10
│   ├── volumes: {medium: Memory, sizeLimit: 2Gi}  ← /dev/shm
│   └── resources: {cpu: 4, memory: 4Gi}
│
└── simulator (Job) — only for testing
```

**Key Kubernetes manifest snippets:**

```yaml
# inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
  namespace: eaigle
spec:
  replicas: 2
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      containers:
        - name: inference
          image: eaigle/inference:latest
          ports:
            - containerPort: 8001
          env:
            - name: YOLO_MODEL
              value: "yolov8s.pt"
          resources:
            limits:
              nvidia.com/gpu: "1"
              memory: "8Gi"
            requests:
              cpu: "2"
              memory: "4Gi"
---
# inference-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
  namespace: eaigle
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-service
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
---
# pipeline-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pipeline
  namespace: eaigle
spec:
  replicas: 5
  template:
    spec:
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 2Gi
      containers:
        - name: pipeline
          image: eaigle/pipeline:latest
          volumeMounts:
            - name: shm
              mountPath: /dev/shm
          env:
            - name: REDIS_URL
              value: "redis://redis-svc:6379/0"
          resources:
            requests:
              cpu: "4"
              memory: "4Gi"
```

**Scaling strategy:**

| Component | Scale Unit | Trigger |
|---|---|---|
| `pipeline` pods | +1 pod per 10 cameras | Fixed: cameras ÷ 10 |
| `inference-service` | +1 pod | GPU utilization > 70% |
| `redis` | Fixed 3 replicas | HA requirement (Sentinel) |

**Important note:** POSIX shared memory is per-node. The `pipeline` pod (CameraWorker + PreprocessingWorkerPool) must run on a single pod since both stages use the same `/dev/shm`. The shm volume above uses `emptyDir: {medium: Memory}` which is pod-local RAM.

For **multi-node** inference scaling: the inference service is already network-separated (HTTP), so it naturally scales horizontally. Only the preprocessing→dispatcher path needs to be co-located.

---

## 6. Configuration Reference

**`configs/pipeline.yaml`** — all tunable parameters:

```yaml
# ── Ingestion ─────────────────────────────────────────────────────────────────
ingestion:
  backend: opencv          # "opencv" | "deepstream"
  deepstream:
    output_width: 1280
    output_height: 720

cameras:
  streams:
    - id: cam_01
      rtsp_url: rtsp://user:pass@192.168.1.10:554/stream
      target_fps: 10
      width: 1920
      height: 1080
      buffer_size: 1         # cv2 capture buffer (1 = always freshest frame)
      reconnect_delay_s: 2.0
      max_consecutive_failures: 30
      zone: "Gate 1"         # Used in event_description

# ── Preprocessing ─────────────────────────────────────────────────────────────
preprocessing:
  num_workers: 4             # ProcessPoolExecutor size (match CPU cores)
  ops:
    - type: color_convert
      params: {src: BGR, dst: RGB}
    - type: resize
      params: {width: 640, height: 640}
    - type: normalize        # ÷255 → float32 (skip if using YOLO's internal norm)
    # - type: gaussian_denoise
    #   params: {ksize: 3}
    # - type: nlm_denoise
    #   params: {h: 10, template_window_size: 7, search_window_size: 21}

# ── Inference ─────────────────────────────────────────────────────────────────
inference:
  service_url: http://localhost:8001
  batch_size: 8              # Max frames per batch
  batch_timeout_ms: 50       # Max wait before flushing partial batch

# ── Aggregation ───────────────────────────────────────────────────────────────
aggregation:
  nms_iou_threshold: 0.45    # Boxes with IoU > this are merged
  confidence_threshold: 0.40  # Detections below this are dropped
  temporal_window: 5          # Number of frames to consider
  temporal_min_presence: 2    # Min appearances in window to confirm label

# ── Redis ─────────────────────────────────────────────────────────────────────
redis:
  url: redis://localhost:6379/0
  stream_maxlen: 500          # Approximate max messages per stream

# ── Observability ─────────────────────────────────────────────────────────────
observability:
  log_level: INFO
  metrics_port: 9090
```

---

## 7. How to Run

### Quick Start (3 commands)

```bash
# 1. Start Redis
docker run -d --name eaigle-redis -p 6379:6379 redis:7-alpine

# 2. Start inference service
.venv/bin/uvicorn inference_service.app:app --port 8001

# 3. Run simulator + pipeline (two terminals)
.venv/bin/python scripts/simulate_cameras.py --cameras 5 --fps 10
.venv/bin/python -m eaigle.app --config configs/pipeline.yaml
```

### Test Inference Service Directly

```bash
# Open API docs in browser
open http://localhost:8001/docs

# Or test from terminal
.venv/bin/python - <<'EOF'
import requests, numpy as np, base64, uuid, cv2

img = cv2.imread("/tmp/sample_frame.jpg")
img = cv2.resize(img, (640, 640))

resp = requests.post("http://localhost:8001/detect/primary", json={
    "batch_id": str(uuid.uuid4()),
    "frames": [{
        "frame_id": str(uuid.uuid4()),
        "camera_id": "test_cam",
        "data_b64": base64.b64encode(img.tobytes()).decode(),
        "shape": list(img.shape),
        "dtype": "uint8",
    }]
})
for det in resp.json()["results"]:
    print(f"{det['label']:12s} conf={det['confidence']:.2f}  bbox={det['bbox']}")
EOF
```

### Run Unit Tests

```bash
.venv/bin/pytest tests/unit/ -v
# 31 tests: preprocessing, batcher, fusion, models, shm
```

### Switch to Larger YOLO Model

```bash
YOLO_MODEL=yolov8s.pt .venv/bin/uvicorn inference_service.app:app --port 8001
```

### Use Real RTSP Camera

Edit `configs/pipeline.yaml`:
```yaml
cameras:
  streams:
    - id: cam_01
      rtsp_url: rtsp://admin:password@192.168.1.100:554/live/ch0
      target_fps: 15
      zone: "Entrance"
```

---

## File Structure

```
EAIGLE_AI/
├── src/eaigle/
│   ├── app.py                          # Main entry point
│   ├── ingestion/
│   │   ├── camera_worker.py            # Single RTSP camera coroutine
│   │   ├── camera_manager.py           # Supervises N cameras
│   │   └── deepstream_manager.py       # DeepStream GPU backend
│   ├── preprocessing/
│   │   ├── shm_utils.py                # POSIX shared memory helpers
│   │   ├── pipeline.py                 # Composable op chain
│   │   ├── worker_pool.py              # ProcessPoolExecutor consumer
│   │   └── ops/                        # Individual preprocessing ops
│   ├── inference/
│   │   ├── batcher.py                  # DynamicBatcher
│   │   └── dispatcher.py              # Reads shm, calls inference API
│   ├── aggregation/
│   │   ├── result_collector.py         # Groups Stage A + B per frame
│   │   ├── hypothesis_fusion.py        # Orchestrates fusion pipeline
│   │   ├── hypothesis_store.py         # Redis persistence + PubSub
│   │   └── fusion_strategies/
│   │       ├── spatial_nms.py          # IoU-based deduplication
│   │       ├── confidence_voting.py    # Score threshold filter
│   │       └── temporal_smoother.py   # 5-frame sliding window
│   ├── models/
│   │   ├── frame.py                    # FrameMetadata dataclass
│   │   ├── detection.py                # BoundingBox, Detection
│   │   └── hypothesis.py              # Hypothesis, CameraConfig
│   ├── transport/
│   │   ├── redis_client.py             # Async Redis connection pool
│   │   ├── stream_producer.py          # XADD wrapper
│   │   └── stream_consumer.py          # XREADGROUP async iterator
│   └── observability/
│       ├── metrics.py                  # Prometheus metrics definitions
│       └── logging_config.py          # Structured JSON logging
├── inference_service/
│   └── app.py                          # FastAPI: YOLOv8 + OCR stub
├── configs/
│   ├── pipeline.yaml                   # Main config
│   └── pipeline-deepstream.yaml        # DeepStream variant config
├── docker/
│   ├── docker-compose.yml              # Full stack
│   ├── Dockerfile.pipeline             # OpenCV backend image
│   ├── Dockerfile.inference            # YOLOv8 inference image
│   ├── Dockerfile.pipeline-deepstream  # DeepStream image
│   └── prometheus.yml                  # Prometheus scrape config
├── scripts/
│   ├── simulate_cameras.py             # Synthetic RTSP simulator
│   └── draw_architecture.py           # Generates docs/architecture.png
├── tests/unit/
│   ├── test_preprocessing.py
│   ├── test_batcher.py
│   ├── test_fusion.py
│   ├── test_models.py
│   └── test_shm.py
├── docs/
│   ├── architecture.png                # Generated diagram
│   ├── architecture.pdf                # Generated diagram (PDF)
│   └── report.md                       # This file
└── pyproject.toml                      # Package definition + dependencies
```

---

*Generated for EAIGLE AI project — 2026-03-04*
