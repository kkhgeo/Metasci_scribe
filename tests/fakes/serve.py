"""A fake server client for the proactive tests."""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock

from langgraph_sdk.errors import NotFoundError

MAIN_GRAPH = {
    "nodes": [],
    "edges": [
        {"source": "__start__", "target": "model"},
        {"source": "model", "target": "HumanInTheLoopMiddleware.after_model"},
        {
            "source": "HumanInTheLoopMiddleware.after_model",
            "target": "Todo.after_model",
        },
        {"source": "Todo.after_model", "target": "tools", "conditional": True},
        {
            "source": "Todo.after_model",
            "target": "memory.after_agent",
            "conditional": True,
        },
        {"source": "tools", "target": "model"},
        {"source": "memory.after_agent", "target": "Interp.after_agent"},
        {"source": "Interp.after_agent", "target": "__end__"},
    ],
}


class FakeCrons:
    def __init__(self):
        self.rows = []
        self.created = []
        self.updated = []
        self._ids = itertools.count(1)

    def search(self, *, metadata, limit, offset):
        rows = [
            r
            for r in self.rows
            if all(r["metadata"].get(k) == v for k, v in metadata.items())
        ]
        return rows[offset : offset + limit]

    def delete(self, cron_id):
        self.rows = [r for r in self.rows if r["cron_id"] != cron_id]

    def create_for_thread(self, thread_id, assistant_id, **kwargs):
        row = {
            "cron_id": f"c{next(self._ids)}",
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            **kwargs,
        }
        self.rows.append(row)
        self.created.append(row)
        return row

    def update(self, cron_id, **kwargs):
        if not any(r["cron_id"] == cron_id for r in self.rows):
            raise NotFoundError("no cron", response=MagicMock(), body=None)
        self.updated.append((cron_id, kwargs))


class FakeThreads:
    def __init__(self):
        self.threads = {}
        self.states = {}
        self.created = []
        self.writes = []
        self.deleted = []

    def get(self, thread_id):
        if thread_id not in self.threads:
            raise NotFoundError("gone", response=MagicMock(), body=None)
        return self.threads[thread_id]

    def get_state(self, thread_id):
        return self.states.get(thread_id, {"values": {}})

    def search(self, *, metadata, limit, sort_by=None, sort_order=None):
        return [
            {"thread_id": tid}
            for tid, t in self.threads.items()
            if all((t.get("metadata") or {}).get(k) == v for k, v in metadata.items())
        ][:limit]

    def delete(self, thread_id):
        self.threads.pop(thread_id, None)
        self.deleted.append(thread_id)

    def create(self, *, metadata):
        tid = f"cron-thread-{len(self.created) + 1}"
        self.threads[tid] = {"thread_id": tid, "metadata": metadata, "status": "idle"}
        self.created.append(tid)
        return {"thread_id": tid}

    def update(self, thread_id, *, metadata):
        self.threads[thread_id].setdefault("metadata", {}).update(metadata)

    def update_state(self, thread_id, values, *, as_node):
        self.writes.append((thread_id, values, as_node))
        return {"checkpoint": {"checkpoint_id": f"after-{len(self.writes)}"}}


class FakeRuns:
    def __init__(self):
        self.running = {}
        self.cancelled = []

    def list(self, thread_id, *, status):
        return self.running.get(thread_id, []) if status == "running" else []

    def cancel(self, thread_id, run_id, *, action):
        self.cancelled.append((thread_id, run_id, action))


class FakeAssistants:
    def __init__(self, graph):
        self.graph = graph

    def get_graph(self, assistant_id):
        assert assistant_id == "metasci-scribe"
        return self.graph


class FakeProactiveClient:
    def __init__(self, graph=MAIN_GRAPH):
        self.crons = FakeCrons()
        self.threads = FakeThreads()
        self.runs = FakeRuns()
        self.assistants = FakeAssistants(graph)


def seed_source_thread(client, tid="t1", *, head="h1", status="idle", interrupts=()):
    client.threads.threads[tid] = {"thread_id": tid, "status": status, "metadata": {}}
    client.threads.states[tid] = {
        "checkpoint": {"checkpoint_id": head},
        "next": [],
        "interrupts": list(interrupts),
    }
