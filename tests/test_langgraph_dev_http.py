"""Smoke tests for the routes mounted via langgraph.json's ``http`` field. We
test the Starlette app directly - no need to spin up langgraph dev.

The only route left is ``/api/bg_process_status``, which the CLI reads across
the process boundary; the app also keeps the browser boundary in front of it.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from metasci_scribe.langgraph_dev.http import app

client = TestClient(app, base_url="http://localhost")


# ---- /api/bg_process_status -----------------------------------------------
# The bg-process reader polls this route (server backend) to detect a process
# exit from the server process's registry.


def test_get_bg_process_status_returns_registry_status(monkeypatch):
    monkeypatch.setattr(
        "metasci_scribe.background.poll_status", lambda process_id: "success"
    )
    resp = client.get("/api/bg_process_status", params={"process_id": "p1"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "success"}


def test_get_bg_process_status_unknown_for_missing_process():
    # Real registry, no such process launched in this test process → unknown.
    resp = client.get("/api/bg_process_status", params={"process_id": "nope"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "unknown"


def test_a_foreign_origin_is_refused():
    resp = client.get(
        "/api/bg_process_status",
        params={"process_id": "p1"},
        headers={"Origin": "https://evil.example"},
    )

    assert resp.status_code == 403


def test_removed_routes_are_gone():
    for path in (
        "/api/models",
        "/api/teams",
        "/api/workspace",
        "/api/policy",
        "/api/settings",
        "/api/bg_processes/running",
        "/api/local-secret/verify",
    ):
        assert client.get(path).status_code == 404, path
