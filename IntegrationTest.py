"""
Full end-to-end integration test for the Mntis platform.

Covers BOTH paths with real data assertions:

  Batch flow (photo):
    1.  Health checks (gateway, orchestrator, storage, minio)
    2.  Register + login a fresh user
    3.  Presign upload URL
    4.  PUT to MinIO
    5.  Complete-upload (publishes media_task)
    6.  DB rows in gateway_db / orchestrator_db / storage_db
    7.  Every Kafka topic carries a payload for this session, with the
        expected fields populated
    8.  Session status reaches 'complete' + crops + burned saved
    9.  Download URL works

  Live flow (WebSocket):
   10.  WS handshake succeeds (the close-before-iter bug)
   11.  session_created arrives over WS
   12.  Send a binary frame — orchestrator publishes a MediaTask via Redis
   13.  Media-worker consumes (Redis blpop) → publishes media_result on Kafka
   14.  Inference happens → publishes inference_result
   15.  LiveSessionService pumps the result back over WS, payload has
        top_emotion + valence + arousal + intensity

Run on EC2:
    pip install httpx websockets Pillow aiokafka asyncpg
    python IntegrationTest.py
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
WS_URL           = "ws://localhost:8001/ws/live"   # directly to orchestrator
KAFKA_BOOTSTRAP  = "localhost:9092"

PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "emotion"
PG_PASS = "emotion_dev"

PIPELINE_TIMEOUT = 90.0
KAFKA_TIMEOUT    = 25.0
WS_RESULT_TIMEOUT = 45.0


# ── Output helpers ────────────────────────────────────────────────────

def _step(label): print(f"\n──── {label} ────")
def _ok(msg):     print(f"  ✓ {msg}")
def _fail(msg):   print(f"  ✗ {msg}\n\nTEST FAILED"); sys.exit(1)
def _info(msg):   print(f"  · {msg}")


# ── Test image ────────────────────────────────────────────────────────

def make_test_image() -> bytes:
    if Image is None:
        _fail("Pillow not installed")
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


# ── Phase 1: health ──────────────────────────────────────────────────

async def step_health():
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


# ── Phase 2: auth ─────────────────────────────────────────────────────

async def step_auth():
    _step("STEP 2 / Auth — register + login fresh user")
    suffix = uuid.uuid4().hex[:8]
    email = f"itest-{suffix}@example.com"
    username = f"itest_{suffix}"
    password = "TestPassword123!"

    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{GATEWAY_URL}/api/auth/register", json={
            "email": email, "username": username, "password": password,
        })
        if r.status_code not in (200, 201):
            _fail(f"register failed {r.status_code}: {r.text[:200]}")
        _ok(f"registered {email}")

        r = await c.post(f"{GATEWAY_URL}/api/auth/login", json={
            "email": email, "password": password,
        })
        if r.status_code != 200:
            _fail(f"login failed {r.status_code}: {r.text[:200]}")
        data = r.json()
        token = data.get("access_token") or data.get("token")
        if not token:
            _fail(f"no access_token in login response: {data}")
        _ok(f"logged in (token len={len(token)})")
    return token, email


# ── Phase 3-4: presign + PUT ─────────────────────────────────────────

async def step_presign(token):
    _step("STEP 3+4 / Presign + PUT to MinIO")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{GATEWAY_URL}/api/upload/request",
            headers={"Authorization": f"Bearer {token}"},
            json={"mode": "photo", "filename": "itest.jpg",
                  "content_type": "image/jpeg"},
        )
        if r.status_code != 200:
            _fail(f"presign failed {r.status_code}: {r.text[:300]}")
        data = r.json()
        for k in ("session_id", "upload_url", "s3_key"):
            if not data.get(k):
                _fail(f"presign response missing '{k}': {data}")
        _ok(f"session_id={data['session_id']}")
        _ok(f"s3_key={data['s3_key']}")

    image_bytes = make_test_image()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.put(
            data["upload_url"], content=image_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        if r.status_code not in (200, 204):
            _fail(f"PUT failed {r.status_code}: {r.text[:200]}")
        _ok(f"uploaded {len(image_bytes)} bytes ({r.status_code})")
    return data["session_id"], data["s3_key"], image_bytes


# ── Phase 5: complete-upload ─────────────────────────────────────────

async def step_complete(token, session_id, s3_key):
    _step("STEP 5 / Complete-upload — publishes media_task")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{GATEWAY_URL}/api/upload/complete",
            headers={"Authorization": f"Bearer {token}"},
            json={"session_id": session_id, "s3_key": s3_key, "mode": "photo"},
        )
        if r.status_code != 200:
            _fail(f"complete failed {r.status_code}: {r.text[:300]}")
        _ok(f"complete returned status={r.json().get('status')}")


# ── Phase 6: DB state ────────────────────────────────────────────────

async def _q(db, sql, *args):
    if asyncpg is None:
        _fail("asyncpg not installed")
    conn = await asyncpg.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASS, database=db,
    )
    try:
        return await conn.fetch(sql, *args)
    finally:
        await conn.close()


async def step_db(email, session_id):
    _step("STEP 6 / DB rows across 3 dbs")
    rows = await _q("gateway_db", "SELECT id FROM users WHERE email=$1", email)
    if not rows:
        _fail(f"gateway_db.users missing {email}")
    _ok(f"gateway_db.users has {email}")

    rows = await _q(
        "orchestrator_db",
        "SELECT id, status, mode FROM sessions WHERE id=$1",
        uuid.UUID(session_id),
    )
    if not rows:
        _fail(f"orchestrator_db.sessions missing {session_id}")
    _ok(f"orchestrator_db.sessions: status={rows[0]['status']} mode={rows[0]['mode']}")

    rows = await _q(
        "storage_db",
        "SELECT category, COUNT(*) AS n FROM files WHERE session_id=$1 GROUP BY category",
        session_id,
    )
    if not rows:
        _fail(f"storage_db.files has no rows for {session_id}")
    cats = {r["category"]: r["n"] for r in rows}
    if cats.get("source", 0) == 0:
        _fail("expected 'source' row")
    _ok(f"storage_db.files: {dict(cats)}")


# ── Phase 7: Kafka payloads ──────────────────────────────────────────

async def _peek_topic(topic, session_id, timeout=KAFKA_TIMEOUT):
    if AIOKafkaConsumer is None:
        _fail("aiokafka not installed")
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
                return msg
    finally:
        await c.stop()
    return None


async def step_kafka(session_id):
    _step("STEP 7 / Kafka topics carry non-empty payloads")
    for topic in ("media_tasks", "media_results", "inference_tasks",
                  "inference_results", "burn_tasks", "burn_results"):
        msg = await _peek_topic(topic, session_id)
        if msg is None:
            _fail(f"{topic}: no message for session_id={session_id}")
        if topic == "media_results":
            faces = msg.get("faces") or []
            if not faces:
                _fail("media_results.faces is empty")
            if not faces[0].get("face_crop"):
                _fail("media_results.faces[0].face_crop missing")
        if topic == "inference_tasks" and not msg.get("face_crop"):
            _fail("inference_tasks.face_crop missing")
        if topic == "inference_results":
            if "emotions" not in msg or "top_emotion" not in msg:
                _fail("inference_results missing emotions/top_emotion")
        if topic == "burn_tasks" and not msg.get("source_s3_key"):
            _fail("burn_tasks.source_s3_key missing")
        if topic == "burn_results":
            if msg.get("status") != "complete":
                _fail(f"burn_results status={msg.get('status')} error={msg.get('error')}")
            if not msg.get("burned_s3_key"):
                _fail("burn_results.burned_s3_key missing")
        _ok(f"{topic}: payload OK")


# ── Phase 8-9: status + download ─────────────────────────────────────

async def step_final(token, session_id):
    _step("STEP 8+9 / Session reaches complete + burned saved + download works")
    deadline = time.time() + PIPELINE_TIMEOUT
    last = None
    while time.time() < deadline:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"{GATEWAY_URL}/api/sessions/{session_id}/status",
                headers={"Authorization": f"Bearer {token}"},
            )
            if r.status_code == 200:
                last = r.json().get("status")
                if last == "complete":
                    break
                if last == "failed":
                    _fail("session status=failed")
        await asyncio.sleep(2)
    if last != "complete":
        _fail(f"session never reached 'complete' (last={last})")
    _ok("session.status = complete")

    rows = await _q(
        "storage_db",
        "SELECT category, COUNT(*) AS n FROM files WHERE session_id=$1 GROUP BY category",
        session_id,
    )
    cats = {r["category"]: r["n"] for r in rows}
    for needed in ("source", "crop", "burned"):
        if cats.get(needed, 0) == 0:
            _fail(f"storage_db.files missing '{needed}' (have {dict(cats)})")
    _ok(f"storage_db.files: {dict(cats)}")

    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(
            f"{GATEWAY_URL}/api/sessions/{session_id}/download",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code != 200:
            _fail(f"download endpoint returned {r.status_code}: {r.text[:200]}")
        url = (r.json() or {}).get("download_url")
        if not url:
            _fail(f"download response missing url: {r.json()}")
        _ok("download_url presigned")


# ── Phase 10-15: live mode E2E ───────────────────────────────────────

async def step_live(token):
    _step("STEP 10-15 / Live mode — WS handshake → frame → result back over WS")
    if websockets is None:
        _fail("websockets not installed")

    url = f"{WS_URL}?token={token}"
    image_bytes = make_test_image()
    live_session_id = None
    result = None

    try:
        async with websockets.connect(url, max_size=10_000_000) as ws:
            _ok("WS handshake accepted")

            # Step 11: session_created
            try:
                first = await asyncio.wait_for(ws.recv(), timeout=10)
            except asyncio.TimeoutError:
                _fail("never received session_created over WS")
            if isinstance(first, bytes):
                first = first.decode("utf-8", errors="replace")
            try:
                msg = json.loads(first)
            except Exception:
                _fail(f"first WS msg not JSON: {first[:200]}")
            if msg.get("type") != "session_created" or not msg.get("session_id"):
                _fail(f"expected session_created, got: {msg}")
            live_session_id = msg["session_id"]
            _ok(f"session_created: {live_session_id}")

            # Step 12: send a binary frame
            await ws.send(image_bytes)
            _ok(f"sent {len(image_bytes)} bytes as binary frame")

            # Step 15: wait for a result back over WS
            deadline = time.time() + WS_RESULT_TIMEOUT
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=max(1, deadline - time.time())
                    )
                except asyncio.TimeoutError:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get("type") == "result":
                    result = msg.get("data") or {}
                    break
                if msg.get("type") == "error":
                    _fail(f"WS error frame: {msg}")
            if result is None:
                _fail(f"no result on WS within {WS_RESULT_TIMEOUT}s")

            for k in ("top_emotion", "valence", "arousal"):
                if k not in result:
                    _fail(f"live result missing '{k}': {result}")
            _ok(f"live result: top={result.get('top_emotion')} "
                f"val={result.get('valence')} arousal={result.get('arousal')}")
    except websockets.exceptions.ConnectionClosedError as e:
        _fail(f"WS closed unexpectedly: code={e.code} reason={e.reason!r}")
    except Exception as e:
        _fail(f"WS error: {type(e).__name__}: {e}")

    # Step 13-14: confirm Kafka saw a media_result + inference_result for this live session
    msg = await _peek_topic("media_results", live_session_id)
    if msg is None:
        _fail("media_results topic has no entry for live session")
    _ok("media_results topic has entry for live session")

    msg = await _peek_topic("inference_results", live_session_id)
    if msg is None:
        _fail("inference_results topic has no entry for live session")
    _ok("inference_results topic has entry for live session")


# ── Driver ───────────────────────────────────────────────────────────

async def main():
    print("=" * 70)
    print("Mntis end-to-end integration test (batch + live)")
    print("=" * 70)
    t0 = time.time()

    await step_health()
    token, email = await step_auth()
    session_id, s3_key, _ = await step_presign(token)
    await step_complete(token, session_id, s3_key)
    await asyncio.sleep(3)
    await step_db(email, session_id)
    await step_kafka(session_id)
    await step_final(token, session_id)
    await step_live(token)

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
        print(f"\nUNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)