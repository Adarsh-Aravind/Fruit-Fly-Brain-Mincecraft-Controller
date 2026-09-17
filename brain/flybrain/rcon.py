"""Minimal Minecraft RCON client (for world setup and episode resets)."""
import socket
import struct


class Rcon:
    def __init__(self, host="127.0.0.1", port=25575, password="flybrain", timeout=5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self._id = 0
        if self._request(3, password)[0] == -1:
            raise PermissionError("RCON authentication failed")

    def _request(self, kind, body):
        self._id += 1
        data = struct.pack("<ii", self._id, kind) + body.encode("utf8") + b"\x00\x00"
        self.sock.sendall(struct.pack("<i", len(data)) + data)
        size = struct.unpack("<i", self._recv(4))[0]
        payload = self._recv(size)
        rid, _ = struct.unpack("<ii", payload[:8])
        return rid, payload[8:-2].decode("utf8", "replace")

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("RCON connection closed")
            buf += chunk
        return buf

    def cmd(self, command):
        return self._request(2, command)[1]

    def close(self):
        self.sock.close()
