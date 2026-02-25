from pathlib import Path

import zotero_mcp.mineru_service as ms


def test_service_uses_official_by_default(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    def fake_official(self, file_path: Path, data_id: str):
        assert file_path == pdf
        assert data_id == "d1"
        return "# official"

    def fail_local(*_args, **_kwargs):
        raise AssertionError("local provider should not be called")

    monkeypatch.setattr(ms.MinerUBatchClient, "parse_pdf_to_markdown", fake_official)
    monkeypatch.setattr(ms.MinerULocalApiClient, "parse_pdf_to_markdown", fail_local)

    svc = ms.MinerUService({"enabled": True, "tokens": ["tok"]})
    assert svc.parse_pdf_to_markdown(pdf, "d1") == "# official"


def test_service_fallbacks_from_local_to_official(monkeypatch, tmp_path):
    pdf = tmp_path / "doc2.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    called = {"local": 0, "official": 0}

    def fail_local(self, file_path: Path, data_id: str):
        called["local"] += 1
        raise RuntimeError("local down")

    def ok_official(self, file_path: Path, data_id: str):
        called["official"] += 1
        return "# official-fallback"

    monkeypatch.setattr(ms.MinerULocalApiClient, "parse_pdf_to_markdown", fail_local)
    monkeypatch.setattr(ms.MinerUBatchClient, "parse_pdf_to_markdown", ok_official)

    svc = ms.MinerUService(
        {
            "enabled": True,
            "provider": "local_api",
            "fallback_providers": ["local_api", "official_upload_batch"],
            "local_api": {"enabled": True, "base_url": "http://localhost:8000", "backend": "vlm"},
            "tokens": ["tok"],
        }
    )
    assert svc.parse_pdf_to_markdown(pdf, "d2") == "# official-fallback"
    assert called == {"local": 1, "official": 1}
