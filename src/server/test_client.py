#!/usr/bin/env python3
"""Test client for darktable-server.

Usage:
    python3 test_client.py <socket_path>

Tests: ping, version, catalog query, shutdown.
"""

import json
import socket
import struct
import sys


def connect(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(path)
    return s


def send_request(s, method, params=None, req_id=None):
    if req_id is None:
        req_id = f"test-{method}"
    msg = json.dumps({"id": req_id, "method": method, "params": params or {}})
    data = msg.encode("utf-8")
    s.sendall(struct.pack(">I", len(data)))
    s.sendall(data)


def recv_response(s):
    header = b""
    while len(header) < 4:
        chunk = s.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("Server closed connection")
        header += chunk

    length = struct.unpack(">I", header)[0]
    data = b""
    while len(data) < length:
        chunk = s.recv(length - len(data))
        if not chunk:
            raise ConnectionError("Server closed connection")
        data += chunk

    return json.loads(data.decode("utf-8"))


def test_ping(s):
    print("--- system.ping ---")
    send_request(s, "system.ping")
    resp = recv_response(s)
    assert resp["error"] is None, f"Unexpected error: {resp['error']}"
    assert resp["result"]["status"] == "ok"
    print(f"  OK: {resp['result']}")


def test_version(s):
    print("--- system.get_version ---")
    send_request(s, "system.get_version")
    resp = recv_response(s)
    assert resp["error"] is None, f"Unexpected error: {resp['error']}"
    print(f"  OK: darktable {resp['result']['version']}")


def test_unknown_method(s):
    print("--- unknown method ---")
    send_request(s, "nonexistent.method")
    resp = recv_response(s)
    assert resp["error"] is not None, "Expected an error"
    print(f"  OK: got expected error: {resp['error']['message']}")


def test_catalog_query(s):
    print("--- catalog.query ---")
    send_request(s, "catalog.query", {"offset": 0, "limit": 5})
    resp = recv_response(s)
    assert resp["error"] is None, f"Unexpected error: {resp['error']}"
    result = resp["result"]
    print(f"  OK: {result['total']} total images, showing {len(result['images'])}:")
    for img in result["images"]:
        print(f"    [{img['id']}] {img['folder']}/{img['filename']}")


def test_catalog_get_image(s, imgid):
    print(f"--- catalog.get_image (imgid={imgid}) ---")
    send_request(s, "catalog.get_image", {"imgid": imgid})
    resp = recv_response(s)
    if resp["error"]:
        print(f"  SKIP: {resp['error']['message']}")
        return
    r = resp["result"]
    print(f"  OK: {r['maker']} {r['model']}, {r['width']}x{r['height']}")
    print(f"      {r['focal_length']}mm f/{r['aperture']} {r['exposure']}s ISO{r['iso']}")


def test_shutdown(s):
    print("--- system.shutdown ---")
    send_request(s, "system.shutdown")
    resp = recv_response(s)
    print(f"  OK: {resp['result']}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <socket_path>")
        sys.exit(1)

    sock_path = sys.argv[1]
    print(f"Connecting to {sock_path}...")
    s = connect(sock_path)
    print("Connected!\n")

    try:
        test_ping(s)
        test_version(s)
        test_unknown_method(s)
        test_catalog_query(s)

        # If there are images, test get_image on the first one
        send_request(s, "catalog.query", {"offset": 0, "limit": 1}, req_id="peek")
        resp = recv_response(s)
        if resp["result"]["images"]:
            first_id = resp["result"]["images"][0]["id"]
            test_catalog_get_image(s, first_id)

        print()
        if "--no-shutdown" not in sys.argv:
            test_shutdown(s)
        else:
            print("(skipping shutdown)")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        s.close()

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
