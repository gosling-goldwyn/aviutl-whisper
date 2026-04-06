"""pywebviewアプリケーション管理"""

import logging
import sys
from importlib import resources
from pathlib import Path

import webview

from .api import Api

logger = logging.getLogger(__name__)

# pywebview + pythonnet(WebView2) の Windows Accessibility 再帰バグ回避
sys.setrecursionlimit(500)


class _PywebviewErrorFilter(logging.Filter):
    """pywebviewのAccessibilityObject再帰エラーを抑制するフィルタ。"""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "AccessibilityObject" in msg:
            return False
        if "CoreWebView2 members can only be accessed" in msg:
            return False
        if "CoreWebView2 can only be accessed" in msg:
            return False
        if "__abstractmethods__" in msg:
            return False
        return True


def get_web_dir() -> str:
    """webアセットディレクトリのパスを取得する。

    PyInstallerでバンドルされた場合は _MEIPASS 配下を参照する。
    """
    # PyInstallerバンドル時
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        return str(base / "aviutl_whisper" / "web")

    # 通常実行時
    web_path = resources.files("aviutl_whisper") / "web"
    return str(web_path)


def main():
    """アプリケーションのエントリーポイント。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # pywebviewの既知エラーログを抑制
    for name in ("pywebview", ""):
        log = logging.getLogger(name)
        log.addFilter(_PywebviewErrorFilter())

    logger.info("aviutl-whisper を起動しています...")

    api = Api()

    web_dir = get_web_dir()
    logger.info("Webアセットディレクトリ: %s", web_dir)

    window = webview.create_window(
        title="aviutl-whisper",
        url=f"{web_dir}/index.html",
        js_api=api,
        width=800,
        height=720,
        min_size=(640, 500),
    )
    api.set_window(window)

    webview.start(
        debug="--debug" in sys.argv,
        http_server=True,
    )


if __name__ == "__main__":
    main()
