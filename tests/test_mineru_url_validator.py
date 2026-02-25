import pytest
from zotero_mcp.mineru_client import _validate_external_https_url, MinerUError


class TestValidateExternalHttpsUrl:
    def test_valid_https_url_passes(self):
        # Should not raise
        _validate_external_https_url("https://mineru.net/api/v4/upload", "upload_url")

    def test_http_scheme_rejected(self):
        with pytest.raises(MinerUError, match="scheme"):
            _validate_external_https_url("http://mineru.net/upload", "upload_url")

    def test_localhost_rejected(self):
        with pytest.raises(MinerUError, match="not allowed"):
            _validate_external_https_url("https://localhost/upload", "upload_url")

    def test_localhost_localdomain_rejected(self):
        with pytest.raises(MinerUError, match="not allowed"):
            _validate_external_https_url("https://localhost.localdomain/upload", "upload_url")

    def test_loopback_ip_rejected(self):
        with pytest.raises(MinerUError, match="non-public"):
            _validate_external_https_url("https://127.0.0.1/upload", "upload_url")

    def test_private_ipv4_rejected(self):
        with pytest.raises(MinerUError, match="non-public"):
            _validate_external_https_url("https://192.168.1.1/upload", "upload_url")

    def test_private_ipv4_10_rejected(self):
        with pytest.raises(MinerUError, match="non-public"):
            _validate_external_https_url("https://10.0.0.1/upload", "upload_url")

    def test_link_local_rejected(self):
        with pytest.raises(MinerUError, match="non-public"):
            _validate_external_https_url("https://169.254.169.254/upload", "upload_url")

    def test_missing_netloc_rejected(self):
        with pytest.raises(MinerUError):
            _validate_external_https_url("https:///upload", "upload_url")

    def test_userinfo_in_netloc_rejected(self):
        with pytest.raises(MinerUError, match="netloc"):
            _validate_external_https_url("https://user@mineru.net/upload", "upload_url")
