"""Capability-based local media reads and atomic, bounded MP4 writes.

No frame rendering or video encoding happens here. The browser owns the muxer.
"""

import base64
import mimetypes
import os
import re
import secrets
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

MAX_CHUNK_BYTES = 256 * 1024


class VideoIO:
    def __init__(self):
        self._lock = threading.RLock()
        self._files = {}
        self._paths = {}
        self._server = None
        self._reservation = None
        self._output = None
        self.busy = False

    def media_url(self, path: str) -> str:
        """Register only an explicit file, never a directory or arbitrary HTTP path."""
        source = Path(path).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"素材がファイルではありません: {source}")
        with self._lock:
            if self._server is None:
                owner = self

                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *args):
                        pass  # URLs contain capabilities; do not log them.

                    def _headers(self):
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range, Accept-Ranges")
                        self.send_header("Cache-Control", "no-store")

                    def do_HEAD(self):
                        self.do_GET(head=True)

                    def do_GET(self, head=False):
                        # A random 256-bit token is the entire read authority.
                        token = urlsplit(self.path).path.removeprefix("/")
                        with owner._lock:
                            file = owner._files.get(token)
                        if file is None:
                            self.send_error(404)
                            return
                        try:
                            with file.open("rb") as stream:
                                size = os.fstat(stream.fileno()).st_size
                                start, end = 0, size - 1
                                requested = self.headers.get("Range")
                                if requested:
                                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                                    if not match or not any(match.groups()):
                                        self.send_error(416)
                                        return
                                    left, right = match.groups()
                                    if left:
                                        start = int(left)
                                        end = min(int(right), end) if right else end
                                    else:
                                        start = max(0, size - int(right))
                                    if start > end or start >= size:
                                        self.send_response(416)
                                        self._headers()
                                        self.send_header("Content-Range", f"bytes */{size}")
                                        self.end_headers()
                                        return
                                self.send_response(206 if requested else 200)
                                self._headers()
                                self.send_header("Accept-Ranges", "bytes")
                                self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
                                self.send_header("Content-Length", str(max(0, end - start + 1)))
                                if requested:
                                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                                self.end_headers()
                                if head:
                                    return
                                stream.seek(start)
                                remaining = end - start + 1
                                while remaining > 0:
                                    block = stream.read(min(64 * 1024, remaining))
                                    if not block:
                                        break
                                    self.wfile.write(block)
                                    remaining -= len(block)
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            pass  # Input.dispose aborts pending range reads.
                        except OSError:
                            self.close_connection = True

                self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                self._server.daemon_threads = True
                threading.Thread(target=self._server.serve_forever, daemon=True).start()
            key = str(source)
            token = self._paths.get(key)
            if token is None:
                token = secrets.token_urlsafe(32)
                self._paths[key] = token
                self._files[token] = source
            return f"http://127.0.0.1:{self._server.server_port}/{token}"

    def reserve(self, path: str) -> dict:
        """Called only with the native SAVE dialog result, never a JS-supplied path."""
        with self._lock:
            if self.busy:
                raise ValueError("動画を書き出し中です")
            destination = Path(path).resolve().with_suffix(".mp4")
            token = secrets.token_urlsafe(32)
            self._reservation = (token, destination)
            return {"token": token, "path": str(destination)}

    def begin(self, token: str):
        with self._lock:
            if self.busy or not self._reservation or token != self._reservation[0]:
                raise ValueError("出力先を選択し直してください")
            _, destination = self._reservation
            fd, temporary = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".mp4.part", dir=destination.parent)
            self._output = (token, destination, Path(temporary), os.fdopen(fd, "wb"))
            self.busy = True

    def write(self, token: str, position: int, encoded: str):
        with self._lock:
            if not self._output or token != self._output[0]:
                raise ValueError("無効な出力セッションです")
            if not isinstance(encoded, str) or len(encoded) > ((MAX_CHUNK_BYTES + 2) // 3) * 4:
                raise ValueError("出力チャンクが大きすぎます")
            data = base64.b64decode(encoded, validate=True)
            if len(data) > MAX_CHUNK_BYTES:
                raise ValueError("出力チャンクが大きすぎます")
            stream = self._output[3]
            if isinstance(position, bool) or not isinstance(position, int) or position != stream.tell():
                raise ValueError("出力チャンクの順序・オフセットが不正です")
            stream.write(data)

    def finish(self, token: str) -> str:
        with self._lock:
            if not self._output or token != self._output[0]:
                raise ValueError("無効な出力セッションです")
            _, destination, temporary, stream = self._output
            if stream.tell() == 0:
                raise ValueError("動画データがありません")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, destination)
            self._output = None
            self._reservation = None
            self.busy = False
            return str(destination)

    def abort(self, token: str | None = None):
        with self._lock:
            if self._output and token is not None and token != self._output[0]:
                raise ValueError("無効な出力セッションです")
            if self._output:
                _, _, temporary, stream = self._output
                try:
                    stream.close()
                    temporary.unlink(missing_ok=True)
                finally:
                    self._output = None
                    self._reservation = None
                    self.busy = False
            if token is None or (self._reservation and token == self._reservation[0]):
                self._reservation = None
            self.busy = False

    def close(self):
        self.abort()
        with self._lock:
            server, self._server = self._server, None
            self._files.clear()
            self._paths.clear()
        if server:
            server.shutdown()
            server.server_close()
