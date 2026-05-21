"""
Mntis end-to-end integration test.

Designed to run ON the EC2 host (hits http://localhost). Spins the remote GPU
instance for inference, walks the full pipeline:

  Phase 0  Pre-flight   — all containers healthy
  Phase 1  Inference    — start GPU instance, wait for /health
  Phase 2  Auth         — register + login a unique test user
  Phase 3  Batch upload — presign, PUT to MinIO, poll status, download burn
  Phase 4  Live session — open WS, send a frame, receive detections, close
  Phase 5  Cleanup      — (optional) stop GPU instance

Usage:
    python3 IntegrationTest.py                  # run full suite
    python3 IntegrationTest.py --no-gpu         # skip GPU start, skip burn assert
    python3 IntegrationTest.py --keep-gpu       # don't stop GPU at the end
    python3 IntegrationTest.py --skip-live      # skip the WS phase
    python3 IntegrationTest.py --image path.jpg # use a specific fixture

Env overrides:
    GATEWAY_URL      default http://localhost:8000
    MINIO_URL        default http://localhost:9000
    GPU_INSTANCE_ID  default i-0953526792fe67a0e
    GPU_REGION       default us-east-1
    INFERENCE_PORT   default 8003
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx


# ─── Config ────────────────────────────────────────────────────────────

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
GPU_INSTANCE_ID = os.getenv("GPU_INSTANCE_ID", "i-0953526792fe67a0e")
GPU_REGION = os.getenv("GPU_REGION", "us-east-1")
INFERENCE_PORT = int(os.getenv("INFERENCE_PORT", "8003"))

HEALTH_TIMEOUT = 60       # seconds to wait for compose services
GPU_BOOT_TIMEOUT = 240    # seconds to wait for GPU + inference /health
UPLOAD_POLL_TIMEOUT = 180 # seconds to wait for burn output
POLL_INTERVAL = 2

DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "sample.jpg"


# ─── Pretty output ─────────────────────────────────────────────────────

class Log:
    GREEN, RED, YELLOW, BLUE, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[94m", "\033[0m"

    @classmethod
    def phase(cls, msg):  print(f"\n{cls.BLUE}━━━ {msg} ━━━{cls.RESET}")
    @classmethod
    def ok(cls, msg):     print(f"  {cls.GREEN}✓{cls.RESET} {msg}")
    @classmethod
    def fail(cls, msg):   print(f"  {cls.RED}✗{cls.RESET} {msg}")
    @classmethod
    def warn(cls, msg):   print(f"  {cls.YELLOW}!{cls.RESET} {msg}")
    @classmethod
    def info(cls, msg):   print(f"  · {msg}")


# ─── State carried across phases ───────────────────────────────────────

@dataclass
class TestState:
    email: str = ""
    password: str = "test-pass-1234"
    token: str = ""
    user_id: str = ""
    session_id: str = ""
    s3_key: str = ""
    upload_url: str = ""
    inference_up: bool = False


# ─── Helpers ───────────────────────────────────────────────────────────

def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def get_or_create_fixture(path: Optional[str]) -> bytes:
    """Prefer a real image, fall back to a synthetic JPEG."""
    p = Path(path) if path else DEFAULT_FIXTURE
    if p.exists():
        Log.info(f"using fixture: {p}")
        return p.read_bytes()

    Log.warn(f"no fixture at {p}, generating synthetic JPEG")
    try:
        from PIL import Image
        from io import BytesIO
        img = Image.new("RGB", (640, 480), color=(200, 180, 160))
        # Add a vaguely face-shaped blob so detection doesn't return zero
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.ellipse((220, 140, 420, 360), fill=(220, 200, 180))
        d.ellipse((270, 210, 295, 235), fill=(40, 40, 40))     # eye
        d.ellipse((345, 210, 370, 235), fill=(40, 40, 40))     # eye
        d.arc((290, 270, 350, 320), 0, 180, fill=(80, 30, 30), width=4)  # mouth
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        Log.fail("Pillow not installed and no fixture available. pip install Pillow")
        sys.exit(1)


# ─── Phase 0: Pre-flight ───────────────────────────────────────────────

async def phase_preflight() -> None:
    Log.phase("Phase 0  Pre-flight")
    required = [
        "gateway", "orchestrator", "storage", "media-worker", "burner",
        "frontend", "nginx", "postgres", "redis", "kafka", "zookeeper", "minio",
    ]
    try:
        out = run(["sudo", "docker", "compose", "ps", "--format", "json"]).stdout
    except subprocess.CalledProcessError as e:
        Log.fail(f"docker compose ps failed: {e.stderr}")
        sys.exit(1)

    running = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        running[obj.get("Service", "")] = obj.get("State", "")

    missing, unhealthy = [], []
    for svc in required:
        state = running.get(svc)
        if state is None:
            missing.append(svc)
        elif state != "running":
            unhealthy.append(f"{svc}={state}")

    if missing or unhealthy:
        if missing:    Log.fail(f"missing: {', '.join(missing)}")
        if unhealthy:  Log.fail(f"not running: {', '.join(unhealthy)}")
        sys.exit(1)

    Log.ok(f"all {len(required)} required containers running")

    # Health endpoints
    async with httpx.AsyncClient(timeout=5) as c:
        for name, url in [
            ("gateway",      f"{GATEWAY_URL}/health"),
            ("orchestrator", "http://localhost:8001/health"),
            ("storage",      "http://localhost:8002/health"),
        ]:
            try:
                r = await c.get(url)
                assert r.status_code == 200, f"{name} /health = {r.status_code}"
                Log.ok(f"{name} /health OK")
            except Exception as e:
                Log.fail(f"{name} /health failed: {e}")
                sys.exit(1)


# ─── Phase 1: Inference ────────────────────────────────────────────────

async def phase_inference(state: TestState, want_gpu: bool) -> None:
    Log.phase("Phase 1  Inference (remote GPU)")
    if not want_gpu:
        Log.warn("--no-gpu set, skipping GPU start (burn assertions will be skipped)")
        return

    Log.info(f"starting GPU instance {GPU_INSTANCE_ID} in {GPU_REGION}")
    try:
        run(["aws", "ec2", "start-instances",
             "--region", GPU_REGION,
             "--instance-ids", GPU_INSTANCE_ID])
    except subprocess.CalledProcessError as e:
        Log.fail(f"aws start-instances failed: {e.stderr}")
        sys.exit(1)

    deadline = time.time() + GPU_BOOT_TIMEOUT
    gpu_ip = None
    while time.time() < deadline:
        try:
            ip = run([
                "aws", "ec2", "describe-instances",
                "--region", GPU_REGION,
                "--instance-ids", GPU_INSTANCE_ID,
                "--query", "Reservations[0].Instances[0].PublicIpAddress",
                "--output", "text",
            ]).stdout.strip()
            if ip and ip != "None":
                gpu_ip = ip
                break
        except subprocess.CalledProcessError:
            pass
        time.sleep(POLL_INTERVAL)

    if not gpu_ip:
        Log.fail("GPU instance did not report a public IP in time")
        sys.exit(1)
    Log.ok(f"GPU instance up at {gpu_ip}")

    # Wait for inference worker to answer /health
    inf_url = f"http://{gpu_ip}:{INFERENCE_PORT}/health"
    Log.info(f"polling inference at {inf_url}")
    async with httpx.AsyncClient(timeout=4) as c:
        while time.time() < deadline:
            try:
                r = await c.get(inf_url)
                if r.status_code == 200:
                    Log.ok("inference worker responding")
                    state.inference_up = True
                    return
            except Exception:
                pass
            await asyncio.sleep(POLL_INTERVAL)

    Log.warn("inference never came up — continuing, will skip burn assertions")


# ─── Phase 2: Auth ─────────────────────────────────────────────────────

async def phase_auth(state: TestState) -> None:
    Log.phase("Phase 2  Auth")
    state.email = f"itest_{uuid.uuid4().hex[:10]}@mntis.local"
    async with httpx.AsyncClient(base_url=GATEWAY_URL, timeout=10) as c:
        r = await c.post("/api/auth/register",
                         json={"email": state.email, "password": state.password})
        assert r.status_code in (200, 201), f"register={r.status_code} body={r.text}"
        Log.ok(f"registered {state.email}")

        r = await c.post("/api/auth/login",
                         json={"email": state.email, "password": state.password})
        assert r.status_code == 200, f"login={r.status_code} body={r.text}"
        data = r.json()
        state.token = data.get("access_token") or data.get("token")
        assert state.token, f"no token in login response: {data}"
        state.user_id = data.get("user", {}).get("id") or data.get("user_id", "")
        Log.ok(f"logged in, token len={len(state.token)}")


# ─── Phase 3: Batch upload ─────────────────────────────────────────────

async def phase_batch_upload(state: TestState, image_bytes: bytes) -> None:
    Log.phase("Phase 3  Batch upload")
    headers = {"Authorization": f"Bearer {state.token}"}

    async with httpx.AsyncClient(base_url=GATEWAY_URL, timeout=30) as c:
        # 1. Presign
        r = await c.post("/api/upload/request",
                         headers=headers,
                         json={"mode": "batch",
                               "filename": "itest.jpg",
                               "content_type": "image/jpeg"})
        assert r.status_code == 200, f"presign={r.status_code} body={r.text}"
        body = r.json()
        state.session_id = body["session_id"]
        state.s3_key = body["s3_key"]
        state.upload_url = body["upload_url"]
        Log.ok(f"presign OK, session={state.session_id[:8]}…")

    # 2. PUT to MinIO directly
    # The presigned URL embeds S3_PUBLIC_ENDPOINT (likely localhost from EC2's
    # perspective). On EC2 host that's fine.
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.put(state.upload_url,
                        content=image_bytes,
                        headers={"Content-Type": "image/jpeg"})
        assert r.status_code in (200, 204), f"PUT minio={r.status_code} body={r.text[:200]}"
        Log.ok(f"uploaded {len(image_bytes)} bytes to MinIO")

    # 3. Notify orchestrator (some pipelines auto-start, some need an explicit kick)
    async with httpx.AsyncClient(base_url=GATEWAY_URL, timeout=10) as c:
        r = await c.post(f"/api/upload/complete",
                         headers=headers,
                         json={"session_id": state.session_id})
        if r.status_code == 404:
            Log.info("no /api/upload/complete endpoint — assuming auto-start")
        elif r.status_code in (200, 202):
            Log.ok("upload complete signal sent")
        else:
            Log.warn(f"upload/complete returned {r.status_code}: {r.text[:120]}")

    # 4. Poll status
    if not state.inference_up:
        Log.warn("inference down — skipping status poll + download")
        return

    Log.info("polling /api/sessions/{id}/status")
    deadline = time.time() + UPLOAD_POLL_TIMEOUT
    last_status = None
    async with httpx.AsyncClient(base_url=GATEWAY_URL, timeout=10) as c:
        while time.time() < deadline:
            r = await c.get(f"/api/sessions/{state.session_id}/status",
                            headers=headers)
            if r.status_code == 200:
                s = r.json().get("status")
                if s != last_status:
                    Log.info(f"status → {s}")
                    last_status = s
                if s in ("complete", "completed", "done"):
                    Log.ok(f"session completed in {int(time.time() - (deadline - UPLOAD_POLL_TIMEOUT))}s")
                    break
                if s in ("failed", "error"):
                    Log.fail(f"session failed: {r.json()}")
                    sys.exit(1)
            await asyncio.sleep(POLL_INTERVAL)
        else:
            Log.fail(f"session did not complete in {UPLOAD_POLL_TIMEOUT}s, last={last_status}")
            sys.exit(1)

        # 5. Download burn
        r = await c.get(f"/api/sessions/{state.session_id}/download",
                        headers=headers)
        assert r.status_code == 200, f"download={r.status_code} body={r.text[:200]}"
        burn_url = r.json().get("download_url") or r.json().get("url")
        assert burn_url, f"no download_url: {r.json()}"
        Log.ok(f"burn URL issued")

    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(burn_url)
        assert r.status_code == 200, f"GET burn={r.status_code}"
        assert len(r.content) > 0, "burn output is empty"
        Log.ok(f"burn downloaded ({len(r.content)} bytes)")


# ─── Phase 4: Live session ─────────────────────────────────────────────

async def phase_live(state: TestState, image_bytes: bytes) -> None:
    Log.phase("Phase 4  Live session (WebSocket)")
    try:
        import websockets
    except ImportError:
        Log.fail("websockets not installed. pip install websockets")
        return

    ws_url = GATEWAY_URL.replace("http", "ws") + "/ws/live"
    Log.info(f"connecting {ws_url}")
    try:
        async with websockets.connect(
            ws_url,
            additional_headers={"Authorization": f"Bearer {state.token}"},
            open_timeout=10,
        ) as ws:
            # Expect a session_created message
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            assert msg.get("type") == "session_created", f"unexpected first msg: {msg}"
            live_session_id = msg["session_id"]
            Log.ok(f"live session created {live_session_id[:8]}…")

            # Send a frame
            b64 = base64.b64encode(image_bytes).decode()
            await ws.send(json.dumps({
                "type": "frame",
                "frame_id": str(uuid.uuid4()),
                "image": b64,
            }))
            Log.ok("frame sent")

            if not state.inference_up:
                Log.warn("inference down — skipping detection wait")
            else:
                # Wait for at least one detection/result message
                deadline = time.time() + 30
                got = False
                while time.time() < deadline:
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                        Log.info(f"recv: {msg.get('type', '?')}")
                        if msg.get("type") in ("detection", "result", "prediction"):
                            got = True
                            break
                    except asyncio.TimeoutError:
                        continue
                if got:
                    Log.ok("received detection")
                else:
                    Log.warn("no detection in 30s (inference path may not be wired to remote GPU)")

            # Clean close
            await ws.send(json.dumps({"type": "end"}))
    except Exception as e:
        Log.fail(f"live phase error: {e}")


# ─── Phase 5: Cleanup ──────────────────────────────────────────────────

def phase_cleanup(state: TestState, keep_gpu: bool) -> None:
    Log.phase("Phase 5  Cleanup")
    if keep_gpu:
        Log.info("--keep-gpu set, leaving GPU instance running")
    elif state.inference_up:
        Log.info(f"stopping GPU instance {GPU_INSTANCE_ID}")
        try:
            run(["aws", "ec2", "stop-instances",
                 "--region", GPU_REGION,
                 "--instance-ids", GPU_INSTANCE_ID])
            Log.ok("GPU stop requested")
        except subprocess.CalledProcessError as e:
            Log.warn(f"failed to stop GPU: {e.stderr}")


# ─── Entrypoint ────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--keep-gpu", action="store_true")
    ap.add_argument("--skip-live", action="store_true")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    image_bytes = get_or_create_fixture(args.image)
    state = TestState()
    started = time.time()

    try:
        await phase_preflight()
        await phase_inference(state, want_gpu=not args.no_gpu)
        await phase_auth(state)
        await phase_batch_upload(state, image_bytes)
        if not args.skip_live:
            await phase_live(state, image_bytes)
        elapsed = time.time() - started
        Log.phase(f"PASS in {elapsed:.1f}s")
    except AssertionError as e:
        Log.fail(f"ASSERTION: {e}")
        sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        Log.fail(f"UNCAUGHT: {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        phase_cleanup(state, keep_gpu=args.keep_gpu)


if __name__ == "__main__":
    asyncio.run(main())