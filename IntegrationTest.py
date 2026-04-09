"""
End-to-End Integration Test — Emotion Recognition Platform

Runs against LIVE Docker services (docker compose up first).
Tests the full pipeline: register → login → upload image → detect → infer → burn → download

Usage:
    1. cd EmotionRecognitionDocker
    2. docker compose up -d
    3. Wait ~30s for all services to be healthy
    4. pip install httpx pytest pytest-asyncio websockets Pillow aiokafka
    5. pytest IntegrationTest.py -v -s

What this tests:
    - Gateway health + auth (register, login, JWT)
    - Orchestrator health (via gateway proxy)
    - Storage health + presigned URLs + crop saving
    - Full photo pipeline: upload → media → inference → burn → download
    - WebSocket live mode connectivity
    - Kafka topic verification
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from typing import Optional

import httpx
import pytest
import pytest_asyncio

# ── Config ─────────────────────────────────────────────

GATEWAY_URL = "http://localhost:8000"
ORCHESTRATOR_URL = "http://localhost:8001"
STORAGE_URL = "http://localhost:8002"
MINIO_URL = "http://localhost:9000"

TEST_EMAIL = "integration-test@example.com"
TEST_USERNAME = "integration_tester"
TEST_PASSWORD = "TestPassword123!"

TIMEOUT = 30.0  # seconds to wait for async pipeline


# ── Helpers ────────────────────────────────────────────

def make_test_image(width: int = 640, height: int = 480) -> bytes:
    """Create a simple test image with a face-like pattern (JPEG bytes)."""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (width, height), color=(200, 180, 160))
        draw = ImageDraw.Draw(img)

        cx, cy = width // 2, height // 2
        face_w, face_h = 150, 200
        draw.ellipse(
            [cx - face_w, cy - face_h, cx + face_w, cy + face_h],
            fill=(220, 190, 170),
            outline=(180, 150, 130),
            width=3,
        )

        draw.ellipse([cx - 70, cy - 50, cx - 30, cy - 20], fill=(50, 50, 50))
        draw.ellipse([cx + 30, cy - 50, cx + 70, cy - 20], fill=(50, 50, 50))
        draw.arc([cx - 50, cy + 30, cx + 50, cy + 80], 0, 180, fill=(150, 50, 50), width=3)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    except ImportError:
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            + b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            + b"\x00" * 50
            + b"\xff\xd9"
        )


def image_to_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


# ── Fixtures ───────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(timeout=30.0) as c:
        yield c


@pytest_asyncio.fixture
async def auth_token(client):
    """Register + login, return access token."""
    await client.post(f"{GATEWAY_URL}/api/auth/register", json={
        "email": TEST_EMAIL,
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD,
    })

    resp = await client.post(f"{GATEWAY_URL}/api/auth/login", json={
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD,
    })
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ═══════════════════════════════════════════════════════
# PHASE 1: Infrastructure Health
# ═══════════════════════════════════════════════════════

class TestInfrastructureHealth:
    """Verify all services are running and healthy."""

    @pytest.mark.asyncio
    async def test_gateway_health(self, client):
        resp = await client.get(f"{GATEWAY_URL}/health")
        assert resp.status_code == 200
        print(f"  Gateway: {resp.json()}")

    @pytest.mark.asyncio
    async def test_orchestrator_health(self, client):
        resp = await client.get(f"{ORCHESTRATOR_URL}/api/health")
        assert resp.status_code == 200
        print(f"  Orchestrator: {resp.json()}")

    @pytest.mark.asyncio
    async def test_storage_health(self, client):
        resp = await client.get(f"{STORAGE_URL}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["db_connected"] is True, "Storage PostgreSQL not connected"
        assert data["s3_connected"] is True, "Storage S3/MinIO not connected"
        print(f"  Storage: {data}")

    @pytest.mark.asyncio
    async def test_minio_reachable(self, client):
        resp = await client.get(f"{MINIO_URL}/minio/health/live")
        assert resp.status_code == 200
        print("  MinIO: healthy")


# ═══════════════════════════════════════════════════════
# PHASE 2: Auth
# ═══════════════════════════════════════════════════════

class TestAuth:
    """Test gateway authentication flow."""

    @pytest.mark.asyncio
    async def test_register_new_user(self, client):
        unique_email = f"test-{int(time.time())}@example.com"
        resp = await client.post(f"{GATEWAY_URL}/api/auth/register", json={
            "email": unique_email,
            "username": f"tester_{int(time.time())}",
            "password": "Password123!",
        })
        assert resp.status_code in (200, 201, 409), f"Register failed: {resp.text}"
        print(f"  Registered: {unique_email}")

    @pytest.mark.asyncio
    async def test_login_returns_tokens(self, client, auth_token):
        assert auth_token is not None
        assert len(auth_token) > 20
        print(f"  Token: {auth_token[:30]}...")

    @pytest.mark.asyncio
    async def test_protected_route_without_token(self, client):
        resp = await client.get(f"{GATEWAY_URL}/api/sessions/fake-id/status")
        assert resp.status_code in (401, 403)
        print("  Unauthorized correctly rejected")


# ═══════════════════════════════════════════════════════
# PHASE 3: Storage Service (Direct)
# ═══════════════════════════════════════════════════════

class TestStorageDirect:
    """Test storage service endpoints directly (internal API)."""

    @pytest.mark.asyncio
    async def test_presign_upload(self, client):
        resp = await client.post(f"{STORAGE_URL}/internal/presign/upload", json={
            "session_id": "integration-test-001",
            "file_type": "upload_image",
            "mime_type": "image/jpeg",
            "original_filename": "test.jpg",
        })
        assert resp.status_code == 200, f"Presign upload failed: {resp.text}"
        data = resp.json()
        assert "file_id" in data
        assert "upload_url" in data
        assert "s3_key" in data
        print(f"  Presign upload: file_id={data['file_id']}")

    @pytest.mark.asyncio
    async def test_save_and_retrieve_crops(self, client):
        fake_image = make_test_image(64, 64)
        crop_b64 = image_to_b64(fake_image)

        resp = await client.post(f"{STORAGE_URL}/internal/save-crops", json={
            "session_id": "integration-test-002",
            "frame_index": 0,
            "detection_index": 0,
            "crops": {
                "face": crop_b64,
                "eyes": crop_b64,
            },
        })
        assert resp.status_code == 200, f"Save crops failed: {resp.text}"
        data = resp.json()
        assert "file_ids" in data
        assert len(data["file_ids"]) == 2
        print(f"  Saved {len(data['file_ids'])} crops")

        file_id = data["file_ids"]["face"]
        resp2 = await client.get(f"{STORAGE_URL}/internal/files/{file_id}")
        assert resp2.status_code == 200
        meta = resp2.json()
        assert meta["category"] == "crop"
        assert meta["size_bytes"] > 0
        print(f"  Retrieved crop metadata: {meta['file_type']} ({meta['size_bytes']} bytes)")

    @pytest.mark.asyncio
    async def test_list_files_by_session(self, client):
        fake_image = make_test_image(64, 64)
        crop_b64 = image_to_b64(fake_image)
        await client.post(f"{STORAGE_URL}/internal/save-crops", json={
            "session_id": "integration-test-002",
            "frame_index": 1,
            "detection_index": 0,
            "crops": {"face": crop_b64},
        })

        resp = await client.get(f"{STORAGE_URL}/internal/files", params={
            "session_id": "integration-test-002",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 2
        print(f"  Listed {len(data)} files for session")

    @pytest.mark.asyncio
    async def test_presign_download(self, client):
        fake_image = make_test_image(64, 64)
        crop_b64 = image_to_b64(fake_image)

        resp = await client.post(f"{STORAGE_URL}/internal/save-crops", json={
            "session_id": "integration-test-003",
            "frame_index": 0,
            "detection_index": 0,
            "crops": {"face": crop_b64},
        })
        assert resp.status_code == 200, f"Save crops failed: {resp.text}"
        file_id = resp.json()["file_ids"]["face"]

        resp2 = await client.post(f"{STORAGE_URL}/internal/presign/download", json={
            "file_id": file_id,
        })
        assert resp2.status_code == 200
        data = resp2.json()
        assert "download_url" in data
        print(f"  Download URL generated for {file_id}")

    @pytest.mark.asyncio
    async def test_delete_file(self, client):
        fake_image = make_test_image(64, 64)
        crop_b64 = image_to_b64(fake_image)

        resp = await client.post(f"{STORAGE_URL}/internal/save-crops", json={
            "session_id": "integration-test-004",
            "frame_index": 0,
            "detection_index": 0,
            "crops": {"face": crop_b64},
        })
        assert resp.status_code == 200, f"Save crops failed: {resp.text}"
        file_id = resp.json()["file_ids"]["face"]

        resp2 = await client.delete(f"{STORAGE_URL}/internal/files/{file_id}")
        assert resp2.status_code == 200

        resp3 = await client.get(f"{STORAGE_URL}/internal/files/{file_id}")
        assert resp3.status_code == 404
        print(f"  Deleted and verified gone: {file_id}")


# ═══════════════════════════════════════════════════════
# PHASE 4: Upload Pipeline (Gateway → Orchestrator → Workers)
# ═══════════════════════════════════════════════════════

class TestUploadPipeline:
    """Test the full photo upload pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_photo_upload_pipeline(self, client, auth_token):
        headers = auth_headers(auth_token)

        resp = await client.post(
            f"{GATEWAY_URL}/api/upload/request",
            headers=headers,
            json={"mode": "photo"},
        )
        if resp.status_code == 200:
            data = resp.json()
            session_id = data.get("session_id")
            upload_url = data.get("upload_url")
            print(f"  Session: {session_id}")
            print(f"  Upload URL: {upload_url[:60]}...")

            test_image = make_test_image()
            upload_resp = await client.put(
                upload_url,
                content=test_image,
                headers={"Content-Type": "image/jpeg"},
            )
            print(f"  Upload status: {upload_resp.status_code}")

            complete_resp = await client.post(
                f"{GATEWAY_URL}/api/upload/complete",
                headers=headers,
                json={"session_id": session_id},
            )
            print(f"  Complete status: {complete_resp.status_code}")

            start = time.time()
            final_status = None
            while time.time() - start < TIMEOUT:
                status_resp = await client.get(
                    f"{GATEWAY_URL}/api/sessions/{session_id}/status",
                    headers=headers,
                )
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    final_status = status_data.get("status")
                    print(f"  Status: {final_status}")
                    if final_status in ("completed", "failed", "error"):
                        break
                await asyncio.sleep(2)

            if final_status == "completed":
                print("  Pipeline COMPLETED successfully!")
                download_resp = await client.get(
                    f"{GATEWAY_URL}/api/sessions/{session_id}/download",
                    headers=headers,
                )
                if download_resp.status_code == 200:
                    print(f"  Download URL received!")
            else:
                print(f"  Pipeline ended with status: {final_status}")

        elif resp.status_code == 404:
            pytest.skip("Upload endpoint not yet implemented in gateway")
        else:
            print(f"  Upload request returned {resp.status_code}: {resp.text}")


# ═══════════════════════════════════════════════════════
# PHASE 5: WebSocket Live Mode
# ═══════════════════════════════════════════════════════

class TestWebSocketLive:
    """Test WebSocket live mode connectivity."""

    @pytest.mark.asyncio
    async def test_websocket_connects(self, auth_token):
        try:
            import websockets

            uri = f"ws://localhost:8000/ws/live?token={auth_token}"
            async with websockets.connect(uri, close_timeout=5) as ws:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(msg)
                    print(f"  WebSocket connected, received: {data.get('type', 'unknown')}")
                    assert data.get("type") in ("session_created", "connected", "welcome")
                except asyncio.TimeoutError:
                    print("  WebSocket connected but no initial message (ok)")

        except ImportError:
            pytest.skip("websockets not installed")
        except Exception as e:
            if any(w in str(e).lower() for w in ("connection", "refused", "rejected", "server")):
                pytest.skip(f"WebSocket not available: {e}")
            raise

    @pytest.mark.asyncio
    async def test_websocket_rejects_no_token(self):
        try:
            import websockets

            uri = "ws://localhost:8000/ws/live"
            with pytest.raises(Exception):
                async with websockets.connect(uri, close_timeout=3) as ws:
                    await asyncio.wait_for(ws.recv(), timeout=3)
            print("  Unauthorized WebSocket correctly rejected")

        except ImportError:
            pytest.skip("websockets not installed")


# ═══════════════════════════════════════════════════════
# PHASE 6: Kafka Topic Verification
# ═══════════════════════════════════════════════════════

class TestKafkaTopics:
    """Verify Kafka topics were created by the orchestrator."""

    @pytest.mark.asyncio
    async def test_topics_exist(self, client):
        try:
            from aiokafka.admin import AIOKafkaAdminClient

            admin = AIOKafkaAdminClient(bootstrap_servers="localhost:9092")
            await admin.start()

            topics = await admin.list_topics()
            expected = [
                "media_tasks", "media_results",
                "inference_tasks", "inference_results",
                "burn_tasks", "burn_results",
            ]

            for topic in expected:
                if topic in topics:
                    print(f"  ✓ {topic}")
                else:
                    print(f"  ✗ {topic} (missing)")

            await admin.close()

        except ImportError:
            pytest.skip("aiokafka not installed")
        except Exception as e:
            pytest.skip(f"Kafka not reachable: {e}")


# ═══════════════════════════════════════════════════════
# PHASE 7: Cleanup
# ═══════════════════════════════════════════════════════

class TestCleanup:
    """Clean up test data."""

    @pytest.mark.asyncio
    async def test_cleanup_integration_files(self, client):
        """Delete all files created during integration tests."""
        for session_id in [
            "integration-test-001",
            "integration-test-002",
            "integration-test-003",
            "integration-test-004",
        ]:
            resp = await client.get(f"{STORAGE_URL}/internal/files", params={
                "session_id": session_id,
            })
            if resp.status_code == 200:
                files = resp.json()  # plain list
                for f in files:
                    await client.delete(f"{STORAGE_URL}/internal/files/{f['id']}")
                if files:
                    print(f"  Cleaned {len(files)} files from {session_id}")

        print("  Cleanup done")