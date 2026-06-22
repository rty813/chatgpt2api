"""本地文件托管：把字节落到 data/hosted 下并通过 /hosted/ 提供下载，替代外部 transfer 服务。

用途：深度研究导出的 PDF、代码解释器产物（sandbox:/mnt/data/...）等需要给用户一个
可点击下载链接的场景。直接由本服务自己托管，不再依赖外部网盘/中转站。
（注意：与 editable_file_task_service 的 data/files（PPT/PSD 产物）刻意分开，互不干扰。）
"""
from __future__ import annotations

import hashlib
import mimetypes
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import FileResponse

from services.config import config

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}


@dataclass(frozen=True)
class StoredFile:
    rel: str
    url: str
    size: int


def _safe_filename(filename: str) -> str:
    """清洗文件名：保留 Unicode（如中文）与扩展名，仅去掉路径分隔符/控制符/文件系统保留字符。"""
    name = str(filename or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)        # 控制字符
    name = re.sub(r"\s+", "_", name)                    # 空白 → 下划线
    name = re.sub(r'[<>:"/\\|?*]', "-", name)          # 文件系统/URL 不安全字符
    name = name.strip(". ")                              # 去掉首尾点和空格
    return (name or "file")[:120]


def _safe_relative_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/").lstrip("/")
    if not value:
        raise HTTPException(status_code=404, detail="file not found")
    parts = Path(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise HTTPException(status_code=404, detail="file not found")
    return Path(*parts).as_posix()


class FileStorageService:
    def local_path(self, rel: str) -> Path:
        safe_rel = _safe_relative_path(rel)
        root = config.hosted_dir.resolve()
        path = (root / safe_rel).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="file not found") from exc
        return path

    def _make_relative_path(self, filename: str, data: bytes) -> str:
        digest = hashlib.md5(data).hexdigest()[:8]
        stamp = int(time.time())
        day = time.strftime("%Y/%m/%d")
        return f"{day}/{stamp}_{digest}/{_safe_filename(filename)}"

    def _public_url(self, rel: str, base_url: str | None = None) -> str:
        encoded = "/".join(quote(seg, safe="") for seg in _safe_relative_path(rel).split("/"))
        base = (base_url or config.base_url).rstrip("/")
        return f"{base}/hosted/{encoded}" if base else f"/hosted/{encoded}"

    def save(self, data: bytes, filename: str, base_url: str | None = None) -> StoredFile:
        """落盘并返回 StoredFile（含可下载 URL）。base_url 为空时回落到 config.base_url。"""
        config.cleanup_old_hosted_files()
        rel = self._make_relative_path(filename, data)
        path = self.local_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return StoredFile(rel=rel, url=self._public_url(rel, base_url), size=len(data))

    def get_file_response(self, rel: str) -> FileResponse:
        path = self.local_path(rel)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, headers=dict(_CORS_HEADERS))


file_storage_service = FileStorageService()
