from pathlib import Path

from zotero_mcp.mineru_client import MinerULocalApiClient, MinerULocalApiConfig


class _Resp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_local_api_client_reads_markdown_from_status(monkeypatch, tmp_path):
    calls = {"post": 0, "get": 0}

    def fake_post(*_args, **_kwargs):
        calls["post"] += 1
        return _Resp(200, {"task_id": "t1"})

    def fake_get(*_args, **_kwargs):
        calls["get"] += 1
        return _Resp(200, {"status": "completed", "data": {"content": "# ok"}})

    import zotero_mcp.mineru_client as mineru_client

    monkeypatch.setattr(mineru_client.requests, "post", fake_post)
    monkeypatch.setattr(mineru_client.requests, "get", fake_get)

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    client = MinerULocalApiClient(MinerULocalApiConfig(base_url="http://localhost:8000"))
    text = client.parse_pdf_to_markdown(Path(pdf), "id")
    assert text == "# ok"
    assert calls["post"] == 1
    assert calls["get"] >= 1


def test_local_api_client_falls_back_to_data_endpoint(monkeypatch, tmp_path):
    state = {"step": 0}

    def fake_post(*_args, **_kwargs):
        return _Resp(200, {"task_id": "t2"})

    def fake_get(url, *args, **kwargs):
        del args, kwargs
        if url.endswith("/api/v1/tasks/t2"):
            state["step"] += 1
            return _Resp(200, {"status": "completed", "data": {}})
        if url.endswith("/api/v1/tasks/t2/data"):
            return _Resp(200, {"status": "completed", "data": {"md": "# from-data"}})
        raise AssertionError(f"Unexpected URL: {url}")

    import zotero_mcp.mineru_client as mineru_client

    monkeypatch.setattr(mineru_client.requests, "post", fake_post)
    monkeypatch.setattr(mineru_client.requests, "get", fake_get)

    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    client = MinerULocalApiClient(MinerULocalApiConfig(base_url="http://localhost:8000"))
    text = client.parse_pdf_to_markdown(Path(pdf), "id")
    assert text == "# from-data"
    assert state["step"] >= 1
