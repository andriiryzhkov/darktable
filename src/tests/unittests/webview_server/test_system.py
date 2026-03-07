"""
Basic system-level integration tests for darktable-server.

These test the JSON-RPC protocol over Unix socket without requiring
a database or images — just verifying the server starts, accepts
connections, and responds to basic RPC calls.
"""

import os
import stat
import pytest


class TestServerConnection:
    """Test server startup and basic connectivity."""

    def test_ping(self, server):
        """Server responds to system.ping."""
        resp = server.call("system.ping")
        assert "result" in resp
        assert resp.get("error") is None

    def test_unknown_method(self, server):
        """Server returns error for unknown method."""
        resp = server.call("nonexistent.method")
        assert "error" in resp


class TestSocketSecurity:
    """Test socket file permissions."""

    def test_socket_permissions(self, server_socket_path, server):
        """Socket file should be owner-only (0600)."""
        mode = os.stat(server_socket_path).st_mode
        perms = stat.S_IMODE(mode)
        # Owner read+write only
        assert perms & stat.S_IRWXG == 0, "Group should have no permissions"
        assert perms & stat.S_IRWXO == 0, "Others should have no permissions"
        assert perms & stat.S_IRUSR != 0, "Owner should have read"
        assert perms & stat.S_IWUSR != 0, "Owner should have write"


class TestCatalog:
    """Test catalog queries (empty database)."""

    def test_query_empty(self, server):
        """catalog.query on empty database returns empty list."""
        resp = server.call("catalog.query", {"offset": 0, "limit": 10})
        assert "result" in resp
        result = resp["result"]
        assert isinstance(result.get("images", result.get("rows")), list)

    def test_get_filmrolls(self, server):
        """catalog.get_filmrolls returns a list."""
        resp = server.call("catalog.get_filmrolls")
        assert "result" in resp

    def test_get_tags(self, server):
        """catalog.get_tags returns a list."""
        resp = server.call("catalog.get_tags")
        assert "result" in resp
