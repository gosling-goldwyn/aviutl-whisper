"""pywebviewアプリケーション管理"""

import logging
import sys
from importlib import resources
from pathlib import Path

import webview

from .api import Api

logger = logging.getLogger(__name__)


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

    webview.start(debug="--debug" in sys.argv)


if __name__ == "__main__":
    main()
