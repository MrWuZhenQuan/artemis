# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Security regression tests for the Artemis console boundary.

Covers the invariants the no-auth security model depends on:
- secrets never appear in HTTP responses,
- media endpoints cannot read files outside the media allowlist,
- cross-origin browser traffic and DNS-rebinding Hosts are rejected,
- no CORS grants exist,
- lifecycle controls stay loopback-only.
"""

import secrets as py_secrets
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from apps.admin_console.server import app
from artemis.config import TRACES_PATH, WORKSPACE_ROOT


def _client(**transport_kwargs) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app, **transport_kwargs), base_url="http://localhost"
    )


# ---------------------------------------------------------------------------
# Secret exposure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_credentials_endpoint_never_returns_key_material(monkeypatch):
    from artemis.config import settings

    honeytoken = f"sk-honeytoken-{py_secrets.token_hex(16)}"
    monkeypatch.setattr(type(settings), "get_api_key", lambda self, provider: SecretStr(honeytoken))

    async with _client() as ac:
        res = await ac.get("/api/system/credentials")

    assert res.status_code == 200
    assert honeytoken not in res.text
    providers = {entry["name"]: entry["configured"] for entry in res.json()["providers"]}
    assert providers.get("google") is True


@pytest.mark.asyncio
async def test_server_status_omits_lifecycle_token_and_metadata():
    async with _client() as ac:
        res = await ac.get("/api/system/server-status")

    assert res.status_code == 200
    data = res.json()
    assert "metadata" not in data
    assert "lifecycle_token" not in res.text
    assert "cmdline" not in res.text
    assert "current_pid" in data


# ---------------------------------------------------------------------------
# File access boundaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_videos_endpoint_refuses_dotenv_and_non_video_files():
    async with _client() as ac:
        for target in (".env", "pyproject.toml", "artemis/__init__.py"):
            res = await ac.get(f"/videos/{target}")
            assert res.status_code in (403, 404), target
            assert "API_KEY" not in res.text


@pytest.mark.asyncio
async def test_videos_endpoint_refuses_encoded_traversal():
    async with _client() as ac:
        for target in (
            "%2e%2e/%2e%2e/etc/passwd",
            "..%5c..%5cwindows%5cwin.ini",
            "%2e%2e%2f.env",
        ):
            res = await ac.get(f"/videos/{target}")
            assert res.status_code in (403, 404), target


@pytest.mark.asyncio
async def test_videos_endpoint_still_serves_real_recordings():
    TRACES_PATH.mkdir(parents=True, exist_ok=True)
    probe = TRACES_PATH / f"security-probe-{py_secrets.token_hex(4)}.mp4"
    probe.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    try:
        async with _client() as ac:
            res = await ac.get(f"/videos/{probe.name}")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("video/mp4")
    finally:
        probe.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_local_file_endpoint_refuses_non_media_workspace_files():
    async with _client() as ac:
        for target in (
            str(WORKSPACE_ROOT / ".env"),
            str(WORKSPACE_ROOT / "pyproject.toml"),
            "file://" + str(WORKSPACE_ROOT / ".env"),
        ):
            res = await ac.get("/local_file", params={"path": target})
            assert res.status_code in (403, 404), target
            assert "API_KEY" not in res.text


@pytest.mark.asyncio
async def test_spa_route_does_not_leak_files_outside_static_roots():
    async with _client() as ac:
        res = await ac.get("/%2e%2e/%2e%2e/pyproject.toml")

    # The catch-all SPA route must fall back to HTML, never the file content.
    assert "requires-python" not in res.text
    assert res.headers["content-type"].startswith("text/html")


# ---------------------------------------------------------------------------
# Browser boundary: Host, Origin, CORS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dns_rebinding_host_is_rejected():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://attacker.example"
    ) as ac:
        res = await ac.get("/api/system/emulator/status")

    assert res.status_code == 403


@pytest.mark.asyncio
async def test_cross_origin_browser_request_is_rejected_without_cors_grant():
    async with _client() as ac:
        res = await ac.post(
            "/api/system/emulator/dismiss",
            headers={"Origin": "https://attacker.example"},
        )

    assert res.status_code == 403
    assert "access-control-allow-origin" not in {k.lower() for k in res.headers}


@pytest.mark.asyncio
async def test_same_origin_browser_request_passes():
    async with _client() as ac:
        res = await ac.get("/api/system/emulator/status", headers={"Origin": "http://localhost"})

    assert res.status_code == 200


@pytest.mark.asyncio
async def test_null_origin_is_rejected():
    async with _client() as ac:
        res = await ac.get("/api/system/emulator/status", headers={"Origin": "null"})

    assert res.status_code == 403


@pytest.mark.asyncio
async def test_security_headers_present_and_api_responses_uncacheable():
    async with _client() as ac:
        res = await ac.get("/api/system/emulator/status")

    assert res.headers.get("x-content-type-options") == "nosniff"
    assert res.headers.get("referrer-policy") == "no-referrer"
    assert res.headers.get("cache-control") == "no-store"


# ---------------------------------------------------------------------------
# Lifecycle controls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_is_loopback_only():
    with patch("threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        async with AsyncClient(
            transport=ASGITransport(app=app, client=("203.0.113.9", 51000)),
            base_url="http://localhost",
        ) as ac:
            res = await ac.post("/api/system/restart")

        assert res.status_code == 403
        mock_thread.assert_not_called()


# ---------------------------------------------------------------------------
# Readiness endpoint secret scrubbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_endpoint_does_not_leak_api_keys(monkeypatch):
    """Honeytoken test: /readiness must never return raw API keys."""
    from artemis.config import settings

    honeytoken = f"sk-honeytoken-{py_secrets.token_hex(16)}"
    monkeypatch.setattr(type(settings), "get_api_key", lambda self, provider: SecretStr(honeytoken))

    async with _client() as ac:
        res = await ac.get("/api/system/readiness")

    assert res.status_code == 200
    assert honeytoken not in res.text
    # Masked fragments must also not leak
    assert honeytoken[:6] not in res.text
    assert honeytoken[-4:] not in res.text


@pytest.mark.asyncio
async def test_readiness_endpoint_is_loopback_only():
    """Non-loopback requests to /readiness must be rejected."""
    async with AsyncClient(
        transport=ASGITransport(app=app, client=("203.0.113.9", 51000)),
        base_url="http://localhost",
    ) as ac:
        res = await ac.get("/api/system/readiness")

    assert res.status_code == 403


@pytest.mark.asyncio
async def test_readiness_omits_credential_probe_metadata():
    """Credential probe metadata must be empty in /readiness response."""
    async with _client() as ac:
        res = await ac.get("/api/system/readiness")

    assert res.status_code == 200
    data = res.json()
    for probe in data.get("probes", []):
        if probe["id"] in ("gemini_api_key", "vision_ocr_key"):
            assert probe["metadata"] == {}, (
                f"Credential probe {probe['id']} must have empty metadata, got: {probe['metadata']}"
            )


@pytest.mark.asyncio
async def test_readiness_preserves_non_credential_probe_metadata():
    """Non-credential probes (ADB, runtime) keep full metadata for debugging."""
    async with _client() as ac:
        res = await ac.get("/api/system/readiness")

    assert res.status_code == 200
    data = res.json()
    # At least one non-credential probe should exist and retain its metadata dict
    non_cred_probes = [
        p for p in data.get("probes", []) if p["id"] not in ("gemini_api_key", "vision_ocr_key")
    ]
    assert len(non_cred_probes) > 0, "Expected at least one non-credential probe"
    for probe in non_cred_probes:
        assert "metadata" in probe, f"Probe {probe['id']} should have a metadata field"


@pytest.mark.asyncio
async def test_credentials_post_does_not_leak_keys_in_report(monkeypatch):
    """POST /credentials updated report must not leak the key just saved."""
    from artemis.config import settings

    honeytoken = f"sk-honeytoken-{py_secrets.token_hex(16)}"
    monkeypatch.setattr(type(settings), "get_api_key", lambda self, provider: SecretStr(honeytoken))

    async def _fake_validate(*a, **kw):
        return True, "ok"

    monkeypatch.setattr("artemis.utils.credentials_validator.validate_api_key", _fake_validate)

    async with _client() as ac:
        res = await ac.post(
            "/api/system/credentials",
            json={"provider": "anthropic", "api_key": honeytoken, "persist_to_env": False},
        )

    assert res.status_code == 200
    assert honeytoken not in res.text


# ---------------------------------------------------------------------------
# Response model validation (Risk 6: response_model removal)
# ---------------------------------------------------------------------------


def test_readiness_endpoint_declares_response_model():
    """The /readiness endpoint must declare response_model=SystemReadinessReport.

    Without it, FastAPI skips response validation, type checking, and field
    filtering — malformed or extra fields go straight to the client.
    A bare dict[str, Any] return annotation is not enough: it accepts any
    dict and filters nothing.
    """
    from apps.admin_console.routers.system import router
    from artemis.core.diagnostics.schema import SystemReadinessReport

    readiness_route = None
    for route in router.routes:
        if getattr(route, "path", None) == "/api/system/readiness":
            readiness_route = route
            break

    assert readiness_route is not None, "Could not find /readiness route"
    assert readiness_route.response_model is SystemReadinessReport, (
        f"/readiness must declare response_model=SystemReadinessReport, "
        f"got {readiness_route.response_model!r} which does not validate "
        f"the response structure or filter undeclared fields"
    )


@pytest.mark.asyncio
async def test_readiness_filters_undeclared_fields_from_response(monkeypatch):
    """response_model must filter out top-level fields not in SystemReadinessReport.

    Without response_model, any extra field in the dict returned by
    _safe_readiness_dict leaks to the client unfiltered — this is the same
    mechanism that caused the API key leak (extra metadata fields serialized
    to the network).
    """
    from apps.admin_console.routers import system as system_router
    from artemis.core.diagnostics.schema import SystemReadinessReport

    valid_report = SystemReadinessReport(
        overall_ready=True,
        blocker_count=0,
        passed_blocker_count=0,
        probes=[],
        os_type="linux",
        timestamp=1234567890.0,
    )

    async def _fake_run_all(**kwargs):
        return valid_report

    original_safe = system_router._safe_readiness_dict

    def _safe_with_extra(report):
        data = original_safe(report)
        data["_internal_secret_field"] = "must_not_reach_client"
        return data

    monkeypatch.setattr(system_router.readiness_engine, "run_all", _fake_run_all)
    monkeypatch.setattr(system_router, "_safe_readiness_dict", _safe_with_extra)

    async with _client() as ac:
        res = await ac.get("/api/system/readiness")

    assert res.status_code == 200
    assert "_internal_secret_field" not in res.json(), (
        "response_model must filter out fields not in SystemReadinessReport schema"
    )


# ---------------------------------------------------------------------------
# Preset provider/model consistency (Risk 12)
# ---------------------------------------------------------------------------


def test_presets_provider_matches_model_name():
    """Each preset in artemis.jsonc must have a provider that matches its model."""
    from artemis.config.paths import get_config_path
    from artemis.utils.file import load_jsonc

    config_path = get_config_path("artemis.jsonc")
    with open(config_path, encoding="utf-8") as f:
        config = load_jsonc(f)

    presets = config.get("presets", {})
    assert len(presets) > 0, "Expected at least one preset in artemis.jsonc"

    EXPECTED_PROVIDER = {
        "openai-gpt4o": "openai",
        "local-ollama": "ollama",
        "gemini-flagship": "google",
        "gemini-flash": "google",
        "cost-saving": "google",
    }

    for name, preset in presets.items():
        provider = preset.get("provider", "")
        expected = EXPECTED_PROVIDER.get(name)
        if expected:
            assert provider == expected, (
                f"Preset '{name}' has provider '{provider}' but should be '{expected}'"
            )
            fb = preset.get("fallback", {})
            assert fb.get("provider") == expected, (
                f"Preset '{name}' fallback has provider '{fb.get('provider')}' "
                f"but should be '{expected}'"
            )


# ---------------------------------------------------------------------------
# Fallback model mapping (Risk 3+4)
# ---------------------------------------------------------------------------


def test_fallback_models_covers_all_providers():
    """FALLBACK_MODELS must have a model name for every provider used in fallback."""
    from artemis.llm.router import FALLBACK_MODELS, ModelProvider

    for provider in (ModelProvider.ANTHROPIC, ModelProvider.OPENAI, ModelProvider.GOOGLE):
        assert provider in FALLBACK_MODELS, f"Missing fallback model for {provider}"
        model = FALLBACK_MODELS[provider]
        assert isinstance(model, str) and model, f"Empty model name for {provider}"


def test_select_fallback_provider_priority(monkeypatch):
    """Provider selection: ANTHROPIC > OPENAI > GOOGLE based on API key availability."""
    from artemis.llm import router as router_module
    from artemis.llm.router import ModelProvider, select_fallback_provider
    from unittest.mock import MagicMock

    # All keys set → ANTHROPIC wins
    mock = MagicMock()
    mock.ANTHROPIC_API_KEY = SecretStr("sk-test")
    mock.OPENAI_API_KEY = SecretStr("sk-test")
    monkeypatch.setattr(router_module, "settings", mock)
    assert select_fallback_provider() == ModelProvider.ANTHROPIC

    # Only OPENAI → OPENAI
    mock.ANTHROPIC_API_KEY = None
    mock.OPENAI_API_KEY = SecretStr("sk-test")
    assert select_fallback_provider() == ModelProvider.OPENAI

    # No keys → GOOGLE (sensible default)
    mock.ANTHROPIC_API_KEY = None
    mock.OPENAI_API_KEY = None
    assert select_fallback_provider() == ModelProvider.GOOGLE


# ---------------------------------------------------------------------------
# Category-based credential scrubbing (Risk 8)
# ---------------------------------------------------------------------------


def test_safe_readiness_dict_scrubs_by_category_not_id():
    """_safe_readiness_dict must scrub metadata for ANY=ProbeCategory.CREDENTIALS probes) probes,
    not just hardcoded IDs. A new credential probe with an unknown ID must still be scrubbed.
    """
    from apps.admin_console.routers import system as system_router
    from artemis.core.diagnostics.schema import (
        ProbeCategory,
        ProbeResult,
        ProbeStatus,
        SystemReadinessReport,
    )

    honeytoken = f"sk-honeytoken-{py_secrets.token_hex(16)}"
    report = SystemReadinessReport(
        overall_ready=True,
        blocker_count=0,
        passed_blocker_count=0,
        probes=[
            ProbeResult(
                id="unknown_new_credential_probe",
                category=ProbeCategory.CREDENTIALS,
                title="New Secret Probe",
                status=ProbeStatus.PASS,
                summary="configured",
                description="configured",
                metadata={"secret": honeytoken},
            ),
        ],
        os_type="linux",
        timestamp=1234567890.0,
    )

    result = system_router._safe_readiness_dict(report)
    probe = result["probes"][0]
    assert probe["metadata"] == {}, (
        f"Metadata should be scrubbed for category=CREDENTIALS, got: {probe['metadata']}"
    )
    assert honeytoken not in str(result), "Honeytoken must not appear in scrubbed response"


# ---------------------------------------------------------------------------
# Loopback env var bypass for readiness only (Risk 9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_allows_remote_with_env_var(monkeypatch):
    """ARTEMIS_ALLOW_REMOTE_READINESS=true lets non-loopback clients read /readiness."""
    monkeypatch.setenv("ARTEMIS_ALLOW_REMOTE_READINESS", "true")
    async with AsyncClient(
        transport=ASGITransport(app=app, client=("203.0.113.9", 51000)),
        base_url="http://localhost",
    ) as ac:
        res = await ac.get("/api/system/readiness")
    assert res.status_code == 200, "Remote readiness should be allowed with env var"


@pytest.mark.asyncio
async def test_restart_still_loopback_only_with_env_var(monkeypatch):
    """ARTEMIS_ALLOW_REMOTE_READINESS must NOT bypass loopback for /restart."""
    monkeypatch.setenv("ARTEMIS_ALLOW_REMOTE_READINESS", "true")
    with patch("threading.Thread") as mock_thread:
        mock_thread.return_value = MagicMock()
        async with AsyncClient(
            transport=ASGITransport(app=app, client=("203.0.113.9", 51000)),
            base_url="http://localhost",
        ) as ac:
            res = await ac.post("/api/system/restart")
        assert res.status_code == 403, "Restart must remain loopback-only"
        mock_thread.assert_not_called()
