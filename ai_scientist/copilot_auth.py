"""
GitHub Copilot API 认证管理器
- OAuth token → Copilot JWT 交换
- JWT 自动刷新（30分钟有效期）
"""

import os
import time
import threading
import requests
import logging

logger = logging.getLogger("copilot-auth")

_lock = threading.Lock()
_cached_jwt: str = ""
_jwt_expires_at: float = 0.0

COPILOT_API = "https://api.individual.githubcopilot.com"


def get_copilot_jwt() -> str:
    """获取有效的 Copilot JWT，过期前自动刷新"""
    global _cached_jwt, _jwt_expires_at

    with _lock:
        # 还有至少 2 分钟有效期就复用
        if _cached_jwt and time.time() < _jwt_expires_at - 120:
            return _cached_jwt

        oauth_token = os.environ.get("COPILOT_OAUTH_TOKEN", "")
        if not oauth_token:
            raise RuntimeError("COPILOT_OAUTH_TOKEN 未设置，无法获取 Copilot JWT")

        headers = {
            "Authorization": "token " + oauth_token,
            "Accept": "application/json",
            "Editor-Version": "vscode/1.100.0",
            "Editor-Plugin-Version": "copilot/1.388.0",
            "User-Agent": "GithubCopilot/1.388.0",
        }

        r = requests.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        _cached_jwt = data["token"]
        _jwt_expires_at = float(data.get("expires_at", time.time() + 1800))
        logger.info(f"Copilot JWT refreshed, expires at {_jwt_expires_at}")
        return _cached_jwt


def copilot_default_headers() -> dict:
    """Copilot API 所需的额外 HTTP 请求头"""
    return {
        "Editor-Version": "vscode/1.100.0",
        "Copilot-Integration-Id": "vscode-chat",
        "User-Agent": "GithubCopilot/1.388.0",
    }
