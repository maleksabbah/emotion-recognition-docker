"""
Thorough end-to-end integration test for the Mntis platform.

Run on EC2:
    pip install httpx websockets Pillow aiokafka asyncpg --break-system-packages
    python IntegrationTest.py

What it verifies, top to bottom:
  1. All services healthy
  2. Auth (register fresh user → login → JWT)
  3. Gateway → Orchestrator: create session
  4. Gateway → Storage: presign upload URL
  5. Browser PUT to MinIO (using presigned URL)
  6. Gateway → Orchestrator: complete upload (publishes media_task)
  7. Three DBs hold the right rows (users / sessions / files)
  8. Kafka topics carry actual payloads with the right fields:
       media_tasks, media_results (faces+crops), inference_tasks (face_crop b64),
       inference_results (emotions+top_emotion), burn_tasks (source_s3_key),
       burn_results (status=complete, burned_s3_key)
  9. Crops + burned written back to MinIO and storage_db.files
 10. Final session.status == complete + frontend can download burned
 11. Live mode: WS /ws/live accepts binary frames, receives predictions

Every step prints PASS / FAIL with the actual reason. Exits 1 on first failure.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import time
import uuid
from typing import Any, Optional

import httpx

try:
    import websockets
except ImportError:
    websockets = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    from aiokafka import AIOKafkaConsumer
except ImportError:
    AIOKafkaConsumer = None


# ── Config ────────────────────────────────────────────────────────────

GATEWAY_URL      = "http://localhost:8000"
ORCHESTRATOR_URL = "http://localhost:8001"
STORAGE_URL      = "http://localhost:8002"
MINIO_URL        = "http://localhost:9000"
WS_URL           = "ws://localhost:8000/ws/live"
KAFKA_BOOTSTRAP  = "localhost:9092"

PG_HOST  = "localhost"
PG_PORT  = 5432
PG_USER  = "emotion"
PG_PASS  = "emotion_dev"

PIPELINE_TIMEOUT = 90.0


# ── Tiny console formatting ───────────────────────────────────────────

def _step(label: str) -> None:
    print(f"\n──── {label} ────")

def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")

def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    print("\nTEST FAILED")
    sys.exit(1)

def _info(msg: str) -> None:
    print(f"  · {msg}")


# ── Test-image factory ────────────────────────────────────────────────

def make_test_image() -> bytes:
    """Return JPEG bytes of a face-shaped image MTCNN should detect."""
    if Image is None:
        _fail("Pillow not installed — pip install Pillow --break-system-packages")
    img = Image.new("RGB", (400, 400), color=(200, 180, 160))
    d = ImageDraw.Draw(img)
    d.ellipse([100, 80, 300, 320], fill=(220, 190, 170))
    d.ellipse([150, 150, 180, 175], fill=(20, 20, 20))
    d.ellipse([220, 150, 250, 175], fill=(20, 20, 20))
    d.polygon([(200, 180), (185, 230), (215, 230)], fill=(180, 150, 130))
    d.arc([170, 240, 230, 280], 0, 180, fill=(80, 30, 30), width=4)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ── Step 1: health checks ─────────────────────────────────────────────

async def step_health() -> None:
    _step("STEP 1 / Health checks")
    async with httpx.AsyncClient(timeout=10) as c:
        for name, url in [
            ("gateway",      f"{GATEWAY_URL}/health"),
            ("orchestrator", f"{ORCHESTRATOR_URL}/health"),
            ("storage",      f"{STORAGE_URL}/health"),
            ("minio",        f"{MINIO_URL}/minio/health/live"),
        ]:
            try:
                r = await c.get(url)
                if r.status_code in (200, 405):
                    _ok(f"{name} reachable ({r.status_code})")
                else:
                    _fail(f"{name} returned {r.status_code}")
            except Exception as e:
                _fail(f"{name} unreachable: {e}")


# ── Step 2: auth ──────────────────────────────────────────────────────

async def step_auth() -> tuple[str, str, str]:
    _step("STEP 2 / Auth: register fresh user + login")
    suffix = uuid.uuid4().hex[:8]
    email = f"itest-{suffix}@example.com"
    username = f"itest_{suffix}"
    password = "TestPassword123!"

    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{GATEWAY_URL}/api/auth/register", json={
            "email": email,
            "username": username,
            "password": password,
        })
        if r.status_code not in (200, 201):
            _fail(f"register failed {r.status_code}: {r.text[:200]}")
        _ok(f"registered {email}")

        r = await c.post(f"{GATEWAY_URL}/api/auth/login", json={
            "email": email,
            "password": password,
        })
        if r.status_code != 200:
            _fail(f"login failed {r.status_code}: {r.text[:200]}")
        data = r.json()
        token = data.get("access_token") or data.get("token")
        if not token:
            _fail(f"no access_token in login response: {data}")
        user_id = (data.get("user") or {}).get("id", "")
        _ok(f"logged in (token len={len(token)})")

    return token, user_id, email


# ── Step 3+4: create session + presign ────────────────────────────────

async def step_presign(token: str) -> dict[str, Any]:
    _step("STEP 3+4 / Create session + presign upload")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{GATEWAY_URL}/api/upload/request",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "mode": "photo",
                "filename": "itest.jpg",
                "content_type": "image/jpeg",
            },
        )
        if r.status_code != 200:
            _fail(f"presign failed {r.status_code}: {r.text[:300]}")
        data = r.json()
        for k in ("session_id", "upload_url", "s3_key"):
            if not data.get(k):
                _fail(f"presign response missing '{k}': {data}")
        _ok(f"session_id={data['session_id']}")
        _ok(f"s3_key={data['s3_key']}")
        _ok(f"presigned url generated")
    return data


# ── Step 5: PUT to MinIO ──────────────────────────────────────────────

async def step_put_to_minio(upload_url: str, image_bytes: bytes) -> None:
    _step("STEP 5 / Browser PUT to MinIO via presigned URL")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.put(
            upload_url,
            content=image_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        if r.status_code not in (200, 204):
            _fail(f"PUT failed {r.status_code}: {r.text[:200]}")
        _ok(f"uploaded {len(image_bytes)} bytes ({r.status_code})")


# ── Step 6: complete-upload ───────────────────────────────────────────

async def step_complete(token: str, session_id: str, s3_key: str) -> None:
    _step("STEP 6 / Complete-upload (gateway → orchestrator → media_tasks)")
    async with httpx.AsyncClient(timeout=15) as c:
        body = {"session_id": session_id, "s3_key": s3_key, "mode": "photo"}
        r = await c.post(
            f"{GATEWAY_URL}/api/upload/complete",
            headers={"Authorization": f"Bearer {token}"},
            json=body,
        )
        if r.status_code != 200:
            _fail(f"complete failed {r.status_code}: {r.text[:300]}")
        _ok(f"complete-upload returned status={r.json().get('status')}")


# ── Step 7: DB state across 3 DBs ─────────────────────────────────────

async def _query(db: str, sql: str, *args):
    if asyncpg is None:
        _fail("asyncpg not installed — pip install asyncpg --break-system-packages")
    conn = await asyncpg.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, database=db,
    )
    try:
        return await conn.fetch(sql, *args)
    finally:
        await conn.close()


async def step_db_state(email: str, session_id: str) -> None:
    _step("STEP 7 / Database state across 3 DBs")

    rows = await _query("gateway_db", "SELECT id FROM users WHERE email=$1", email)
    if not rows:
        _fail(f"gateway_db.users has no row for {email}")
    _ok(f"gateway_db.users has row for {email}")

    rows = await _query(
        "orchestrator_db",
        "SELECT id, status, mode FROM sessions WHERE id=$1",
        uuid.UUID(session_id),
    )
    if not rows:
        _fail(f"orchestrator_db.sessions missing {session_id}")
    _ok(f"orchestrator_db.sessions: status={rows[0]['status']} mode={rows[0]['mode']}")

    rows = await _query(
        "storage_db",
        "SELECT category, COUNT(*) AS n FROM files WHERE session_id=$1 GROUP BY category",
        session_id,
    )
    if not rows:
        _fail(f"storage_db.files has no rows for session {session_id}")
    cats = {r["category"]: r["n"] for r in rows}
    _ok(f"storage_db.files: {dict(cats)}")
    if cats.get("source", 0) == 0:
        _fail("expected 'source' row in storage_db.files")


# ── Step 8: Kafka payload checks ──────────────────────────────────────

async def _peek_for_session(topic: str, session_id: str, status: Optional[str] = None,
                            timeout: float = 25.0) -> dict | None:
    """Read messages on topic until we find one matching our session_id."""
    if AIOKafkaConsumer is None:
        _fail("aiokafka not installed — pip install aiokafka --break-system-packages")
    c = AIOKafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="earliest",
        group_id=f"itest-{uuid.uuid4().hex[:6]}",
        consumer_timeout_ms=int(timeout * 1000),
    )
    await c.start()
    try:
        async for record in c:
            try:
                msg = json.loads(record.value.decode("utf-8"))
            except Exception:
                continue
            if msg.get("session_id") == session_id:
                if status and msg.get("status") != status:
                    continue
                return msg
    finally:
        await c.stop()
    return None


async def step_kafka_payloads(session_id: str) -> None:
    _step("STEP 8 / Kafka topics carry non-empty payloads for this session")
    for topic in ("media_tasks", "media_results", "inference_tasks",
                  "inference_results", "burn_tasks", "burn_results"):
        msg = await _peek_for_session(topic, session_id)
        if msg is None:
            _fail(f"{topic}: no message for session_id={session_id}")
        if topic == "media_results":
            faces = msg.get("faces") or []
            if not faces:
                _fail("media_results.faces is empty (no face detected)")
            if not faces[0].get("face_crop"):
                _fail("media_results.faces[0].face_crop is empty (no b64 crop)")
        if topic == "inference_tasks":
            if not msg.get("face_crop"):
                _fail("inference_tasks.face_crop missing")
        if topic == "inference_results":
            if "emotions" not in msg or "top_emotion" not in msg:
                _fail("inference_results missing emotions/top_emotion")
        if topic == "burn_tasks":
            if not msg.get("source_s3_key"):
                _fail("burn_tasks.source_s3_key missing")
        if topic == "burn_results":
            if msg.get("status") != "complete":
                _fail(f"burn_results status={msg.get('status')} error={msg.get('error')}")
            if not msg.get("burned_s3_key"):
                _fail("burn_results.burned_s3_key missing")
        _ok(f"{topic}: payload OK")


# ── Step 9-10: final state + download ─────────────────────────────────

async def step_final_state(token: str, session_id: str) -> None:
    _step("STEP 9-10 / Final state: status complete + burned saved + download works")

    deadline = time.time() + PIPELINE_TIMEOUT
    last_status = None
    while time.time() < deadline:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"{GATEWAY_URL}/api/sessions/{session_id}/download",
                headers={"Authorization": f"Bearer {token}"},
            )
            if r.status_code == 200:
                last_status = r.json().get("status")
                if last_status == "complete":
                    break
                if last_status == "failed":
                    _fail("session status=failed")
        await asyncio.sleep(2)
    if last_status != "complete":
        _fail(f"session never reached 'complete' (last_status={last_status})")
    _ok("session.status = complete")

    rows = await _query(
        "storage_db",
        "SELECT category, COUNT(*) AS n FROM files WHERE session_id=$1 GROUP BY category",
        session_id,
    )
    cats = {r["category"]: r["n"] for r in rows}
    for required in ("source", "crop", "burned"):
        if cats.get(required, 0) == 0:
            _fail(f"storage_db.files missing '{required}' (have {dict(cats)})")
    _ok(f"storage_db.files: {dict(cats)}")

    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(
            f"{GATEWAY_URL}/api/download/{session_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code != 200:
            _fail(f"download endpoint returned {r.status_code}: {r.text[:200]}")
        url = (r.json() or {}).get("download_url")
        if not url:
            _fail(f"download response missing url: {r.json()}")
        _ok("download_url presigned")


# ── Step 11: live mode ────────────────────────────────────────────────

async def step_live_mode(token: str) -> None:
    _step("STEP 11 / Live mode: WS /ws/live → media → inference → result")
    if websockets is None:
        _fail("websockets not installed — pip install websockets --break-system-packages")

    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{GATEWAY_URL}/api/sessions",
            headers={"Authorization": f"Bearer {token}"},
            json={"mode": "live"},
        )
        if r.status_code not in (200, 201):
            _fail(f"create live session failed {r.status_code}: {r.text[:200]}")
        body = r.json()
        session_id = body.get("id") or body.get("session_id")
        if not session_id:
            _fail(f"create live session missing id: {body}")
        _ok(f"live session created: {session_id}")

    image_bytes = make_test_image()
    url = f"{WS_URL}?session_id={session_id}&token={token}"
    try:
        async with websockets.connect(url, max_size=10_000_000) as ws:
            _ok("WS connected")
            await ws.send(image_bytes)
            _ok(f"WS sent {len(image_bytes)} binary bytes")
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                _fail("no live result on WS within 30s")
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8", errors="replace")
            data = json.loads(msg) if msg.lstrip().startswith("{") else {"raw": msg}
            for k in ("top_emotion", "valence", "arousal"):
                if k not in data:
                    _fail(f"live result missing '{k}': {data}")
            _ok(f"live result: top={data.get('top_emotion')} "
                f"val={data.get('valence')} arousal={data.get('arousal')}")
    except Exception as e:
        _fail(f"WS error: {e}")


# ── Driver ────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 70)
    print("Mntis end-to-end integration test")
    print("=" * 70)
    t0 = time.time()

    await step_health()
    token, user_id, email = await step_auth()
    presign = await step_presign(token)
    image_bytes = make_test_image()
    await step_put_to_minio(presign["upload_url"], image_bytes)
    await step_complete(token, presign["session_id"], presign["s3_key"])

    await asyncio.sleep(3)

    await step_db_state(email, presign["session_id"])
    await step_kafka_payloads(presign["session_id"])
    await step_final_state(token, presign["session_id"])
    await step_live_mode(token)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"ALL CHECKS PASSED in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)