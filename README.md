**# Mntis — Real-Time Facial Emotion Recognition Platform

Batch and live facial emotion recognition, built on a multi-stream CNN
trained on public emotion datasets and an event-driven microservices
architecture.

Mntis analyses faces — both as batch jobs (upload a photo or video, get
back per-face emotion predictions burned into an annotated image) and
live (stream from your webcam, get emotion + valence + arousal in real
time). It is built as eight independent services communicating over
Kafka, with model inference isolated into its own worker.

Deployed at [mntis.app](https://mntis.app).

## What it does

- **Batch pipeline** — upload a photo (or video frame) → faces are
  detected → for each face, eyes / mouth / cheeks / forehead are
  cropped via MediaPipe landmarks → a multi-stream CNN predicts
  emotion, valence, arousal, and intensity → the source image is
  re-rendered with bounding boxes and labels burned in, ready to
  download.
- **Live recognition** — stream webcam frames over a WebSocket and
  receive incremental emotion predictions in real time, with stability
  smoothing and a top-3 distribution display.
- **Region-aware model** — the network fuses five image streams (the
  whole face plus four landmark-cropped regions) rather than only
  looking at the face as a single image. Eye, mouth, cheek, and
  forehead crops each go through their own CNN branch before fusion.

## Architecture

Mntis is an event-driven system. A thin **Gateway** is the only public
HTTP entrypoint; it authenticates requests and proxies to internal
services. The **Orchestrator** owns session lifecycle and coordinates
the pipeline by publishing tasks to Kafka, which the worker services
consume independently. The model lives in a dedicated **Inference
Worker** that can scale separately from the rest of the stack. The
**Burner** runs at the end of the batch pipeline to produce the final
annotated image.

## Services

| Service          | Repo                              | Responsibility                                                                                          |
| ---------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Gateway          | `EmotionRecognitionGateway`       | Public API. JWT auth, request validation, presigned URL brokering, file download proxy.                 |
| Orchestrator     | `EmotionRecognitionOrchestrator`  | Session lifecycle, pipeline coordination via Kafka, live WebSocket at `/ws/live`.                       |
| Storage          | `EmotionRecognitionStorage`       | File registry + presigned upload/download URLs over MinIO. Per-file metadata in `storage_db`.           |
| Media Worker     | `EmotionRecognitionMedia`         | Face detection (MTCNN), landmark extraction (MediaPipe FaceMesh), region cropping (PIL polygon).        |
| Inference Worker | `EmotionRecognitionInference`     | Model inference. Multi-stream CNN (ResNet18 face + 4 RegionCNNs + attention fusion + 4 prediction heads). |
| Burner           | `EmotionRecognitionBurner`        | Renders bounding boxes and emotion labels onto the source image; uploads the annotated artefact.        |
| Frontend         | `emotion-frontend`                | Next.js UI (upload, sessions archive, live webcam). Served through nginx.                               |
| Docker           | `EmotionRecognitionDocker`        | Compose orchestration, nginx config, certificates, integration test, model checkpoint mount.            |

## Pipeline flow

### Batch (photo → annotated image)

A user uploads a file and gets back a predicted emotion per face,
overlaid on the original image. The steps:

1. **Presign & upload.** The client asks the gateway for a presigned
   URL and uploads the file straight to object storage (MinIO). Large
   files never pass through the API itself.
2. **Session created.** The gateway forwards a complete-upload request
   to the orchestrator, which records the session and publishes a
   `media_task` to Kafka.
3. **Face detection + region crops.** The media worker consumes the
   task, decodes the image with PIL, runs MTCNN to find faces,
   resizes each face to 128×128, then runs MediaPipe FaceMesh to
   extract polygon crops for eyes, mouth, cheeks, and forehead.
   Crops are base64-encoded and a `media_result` is published, and
   the crops are also saved to MinIO under the session.
4. **Inference.** The orchestrator turns each face in the result into
   an `inference_task`. The inference worker consumes it, decodes the
   crops with PIL (matching training preprocessing exactly), runs the
   multi-stream CNN, and publishes `inference_result` carrying the
   top emotion, full per-class probabilities, valence, arousal, and
   intensity.
5. **Burning.** Once every detection in the session has a matching
   prediction, the orchestrator publishes a `burn_task`. The burner
   downloads the original source from MinIO, draws bounding boxes and
   emotion labels onto the image with OpenCV, uploads the annotated
   image back to MinIO, and publishes `burn_result`.
6. **Done.** The orchestrator consumes the burn result, marks the
   session `complete`, and the annotated image is available via a
   presigned download URL.

Every hand-off between workers is a Kafka event rather than a direct
call. A task waits in its topic until the right worker picks it up —
if a worker is briefly down, the work simply queues instead of
failing. The frontend polls the session's status endpoint throughout.

### Live (webcam → real-time emotion)

For live recognition the client opens a WebSocket directly to the
orchestrator (`/ws/live?token=…`, routed through nginx, which proxies
the upgrade to the orchestrator container). The steps:

1. The browser captures webcam frames at ~1.5 FPS, encodes them as
   JPEG via `canvas.toBlob`, and streams them as binary WebSocket
   frames.
2. The orchestrator's `LiveSessionService` caches each frame in Redis
   and enqueues a `MediaTask` whose `frame_source.type = "redis"`
   onto a per-session Redis list.
3. The media worker's Redis loop scans active live sessions,
   `blpop`s the next frame's task, runs the same detection +
   cropping pipeline as batch, and publishes a `media_result`.
4. The orchestrator turns the result into an `inference_task` for
   the inference worker as usual. The resulting `inference_result`
   is identified as a live frame and pushed onto a Redis stream for
   that session, rather than triggering a burn.
5. The orchestrator's per-session pump task reads the stream and
   forwards each prediction back over the WebSocket as
   `{"type": "result", "data": {...}}`. The frontend smooths the
   predictions (3-frame stability + 40% confidence threshold) and
   renders the headline emotion, top-3 distribution bars, and
   valence/arousal/intensity dimensions.
6. When the client disconnects, the session ends and the per-session
   Redis state is cleaned up.

## Tech stack

- **Services:** Python, FastAPI, clean architecture
  (Routes → Services → Repositories → Entities).
- **Messaging:** Apache Kafka, event-driven task distribution. Six
  topics: `media_tasks`, `media_results`, `inference_tasks`,
  `inference_results`, `burn_tasks`, `burn_results`.
- **Model:** Custom multi-stream CNN — ResNet18 face encoder + four
  64×64 region encoders + feature-attention fusion + shared FC +
  four prediction heads (emotion / valence / arousal / intensity).
  Trained on FER+, RAF-DB, AffectNet, ExpW, and CK+ via PyTorch.
- **Detection:** MTCNN for face bounding boxes, MediaPipe FaceMesh
  for landmark-based region polygons. PIL for all image
  manipulation (matches training preprocessing byte-for-byte).
- **Data:** PostgreSQL (per-service databases — `gateway_db`,
  `orchestrator_db`, `storage_db`), Redis (live sessions and
  progress), MinIO (S3-compatible object store).
- **Infra:** Docker Compose, AWS EC2, nginx reverse proxy,
  Let's Encrypt HTTPS.
- **Frontend:** Next.js, served via nginx alongside the API.

## Design decisions

- **Event-driven over direct calls.** Services communicate through
  Kafka topics rather than synchronous chains, so a slow or
  unavailable worker never blocks the rest of the pipeline — tasks
  buffer in the queue.
- **Separate inference worker.** The model is the heaviest dependency
  in the stack (PyTorch + ~50 MB checkpoint). Isolating it means the
  rest of the pipeline can iterate without rebuilding heavy images,
  and inference can be scaled horizontally on its own.
- **Gateway as the only public surface.** Internal services are
  never exposed directly; the gateway authenticates and proxies,
  keeping the trust boundary in one place. The one exception is the
  live-mode WebSocket, which nginx routes straight to the
  orchestrator because that's where the WS handler lives.
- **Per-service databases.** Each service owns its schema, avoiding
  shared-database coupling. SQLAlchemy with async sessions in every
  service.
- **Presigned uploads.** Clients upload directly to object storage
  via presigned URLs, so large files never pass through the API.
  Live frames go via WebSocket + Redis instead, since they're small
  and need to flow through the system quickly.
- **PIL-based pipeline.** Cropping and resizing match the training
  pipeline byte-for-byte (PIL `Image.open` / `.crop` / `.resize` /
  `.save` at JPEG quality 95), because subtle cv2-vs-PIL differences
  in interpolation and color order were producing wrong predictions
  on real photos despite the model loading correctly.**
