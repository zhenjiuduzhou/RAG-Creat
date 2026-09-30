"""Per-user API secrets protected with Windows DPAPI."""
from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
from ctypes import wintypes
from pathlib import Path


def _identity(endpoint: str, model_id: str) -> str:
    return hashlib.sha256(json.dumps([endpoint, model_id], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _crypt(data: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise RuntimeError("API credentials require Windows DPAPI")
    source_buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source = _Blob(len(data), source_buffer)
    destination = _Blob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.argtypes = ([ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.POINTER(_Blob),
                          ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
                         if decrypt else
                         [ctypes.POINTER(_Blob), ctypes.c_wchar_p, ctypes.POINTER(_Blob),
                          ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)])
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(destination)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(destination.pbData, destination.cbData)
    finally:
        local_free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
        local_free.argtypes = [ctypes.c_void_p]
        local_free.restype = ctypes.c_void_p
        local_free(destination.pbData)


class CredentialsStore:
    def __init__(self, home: Path):
        self.path = Path(home) / "credentials.json"

    def lookup(self, endpoint: str | dict, model_id: str | None = None) -> str | None:
        if isinstance(endpoint, dict):
            endpoint, model_id = endpoint["endpoint"], endpoint["model_id"]
        if model_id is None:
            raise ValueError("model_id is required")
        override = os.getenv("RAG_STUDIO_EMBEDDING_KEY")
        if override:
            return override
        if not self.path.is_file():
            return None
        entries = json.loads(self.path.read_text(encoding="utf-8"))
        encoded = entries.get(_identity(endpoint, model_id))
        if encoded is None:
            return None
        return _crypt(base64.b64decode(encoded, validate=True), decrypt=True).decode("utf-8")

    def save(self, endpoint: str | dict, model_id: str, key: str | None = None) -> None:
        if isinstance(endpoint, dict):
            endpoint, model_id, key = endpoint["endpoint"], endpoint["model_id"], model_id
        if not key:
            raise ValueError("API key cannot be empty")
        encrypted = base64.b64encode(_crypt(key.encode("utf-8"), decrypt=False)).decode("ascii")
        entries = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        entries[_identity(endpoint, model_id)] = encrypted
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)


CredentialStore = CredentialsStore
