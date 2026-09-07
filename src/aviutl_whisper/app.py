"""pywebviewアプリケーション管理"""

import logging
import os
import sys
import threading
from importlib import resources
from pathlib import Path

import webview

from .api import Api

logger = logging.getLogger(__name__)

REMOTE_DEBUGGING_PORT_ENV = "AVIUTL_WHISPER_REMOTE_DEBUGGING_PORT"

# pywebview + pythonnet(WebView2) の Windows Accessibility 再帰バグ回避
# 500 では WebView2 の正常な初期化でも RecursionError が発生する
sys.setrecursionlimit(3000)


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


def _configure_remote_debugging() -> None:
    """テスト用CDPポートをWebView2のAPI設定へ反映する。"""
    value = os.environ.get(REMOTE_DEBUGGING_PORT_ENV)
    if value is None:
        return

    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{REMOTE_DEBUGGING_PORT_ENV} must be an integer"
        ) from exc

    if not 1 <= port <= 65535:
        raise ValueError(
            f"{REMOTE_DEBUGGING_PORT_ENV} must be between 1 and 65535"
        )

    # WebView2 Runtime 150以降は昇格プロセスから渡された
    # WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTSを無視するため、pywebviewの
    # CoreWebView2CreationProperties経由でブラウザ引数を設定する。
    webview.settings["REMOTE_DEBUGGING_PORT"] = port
    logger.info("WebView2 CDPポート: %d", port)


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

    _configure_remote_debugging()

    api = Api()

    web_dir = get_web_dir()
    logger.info("Webアセットディレクトリ: %s", web_dir)

    hidden = os.environ.get("AVIUTL_WHISPER_HIDDEN", "0") == "1"

    window = webview.create_window(
        title="aviutl-whisper",
        url=f"{web_dir}/index.html",
        js_api=api,
        width=800,
        height=720,
        min_size=(640, 500),
        hidden=hidden,
    )
    api.set_window(window)

    def on_closing():
        """アプリ終了前に未保存の変更を確認する。

        dirty状態の場合は終了をキャンセルし、JSに3択ダイアログを表示させる。
        """
        if api._skip_close_dialog:
            return  # 強制終了 → 許可

        if not api._is_dirty:
            return  # 変更なし → 許可

        # dirty → 終了をキャンセルし JS 側のダイアログを別スレッドで呼び出す
        # NOTE: on_closing は GUIスレッドで発火するため、evaluate_js を同スレッドで
        #       同期呼び出しするとデッドロックになる。threading.Thread で回避する。
        threading.Thread(
            target=lambda: window.evaluate_js(
                "window._showCloseConfirm && window._showCloseConfirm()"
            ),
            daemon=True,
        ).start()
        return False

    window.events.closing += on_closing

    webview.start(
        debug="--debug" in sys.argv,
        http_server=True,
    )


if __name__ == "__main__":
    main()
