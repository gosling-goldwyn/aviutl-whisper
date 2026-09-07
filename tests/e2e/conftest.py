"""E2E テスト共通フィクスチャ

pywebview (WebView2) アプリを CDP 経由で起動・接続する。
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from tests.e2e.helpers import reset_app_state

PROJECT_ROOT = Path(__file__).parent.parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures"
REMOTE_DEBUG_PORT = 9222
APP_LAUNCH_TIMEOUT = 30


def _stop_process(proc: subprocess.Popen) -> int | None:
    """アプリプロセスを停止し、終了コードを返す。"""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    return proc.returncode


# ---------------------------------------------------------------------------
# セッションスコープ: アプリプロセス・ブラウザ接続
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app_process(tmp_path_factory):
    """WebView2 リモートデバッグポート付きでアプリをサブプロセス起動するフィクスチャ。

    - ``AVIUTL_WHISPER_HIDDEN=1`` で不可視ウィンドウ起動（ヘッドレス相当）
    - アプリ固有の環境変数で settings.json を分離
    - CDPポートはpywebviewのWebView2 API設定を経由して指定
    """
    tmp_settings = tmp_path_factory.mktemp("settings")
    log_path = tmp_settings / "app.log"

    env = os.environ.copy()
    env.pop("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", None)
    env["AVIUTL_WHISPER_REMOTE_DEBUGGING_PORT"] = str(REMOTE_DEBUG_PORT)
    env["AVIUTL_WHISPER_HIDDEN"] = "1"
    env["AVIUTL_WHISPER_SETTINGS_PATH"] = str(tmp_settings / "settings.json")

    python = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    with log_path.open("wb") as app_log:
        proc = subprocess.Popen(
            [python, str(PROJECT_ROOT / "main.py")],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=app_log,
            stderr=subprocess.STDOUT,
        )

        # CDP エンドポイントが利用可能になるまでポーリング
        deadline = time.time() + APP_LAUNCH_TIMEOUT
        connected = False
        last_probe_error: Exception | None = None
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{REMOTE_DEBUG_PORT}/json/version",
                    timeout=1,
                ) as response:
                    response.read()
                connected = True
                break
            except Exception as exc:
                last_probe_error = exc
                time.sleep(0.5)

        if not connected:
            return_code = _stop_process(proc)
            app_log.flush()
            app_output = log_path.read_bytes().decode("utf-8", errors="replace")
            pytest.fail(
                f"App did not expose CDP on port {REMOTE_DEBUG_PORT} "
                f"within {APP_LAUNCH_TIMEOUT}s\n"
                f"Process return code: {return_code}\n"
                f"Last CDP probe error: {last_probe_error!r}\n"
                f"App output:\n{app_output or '<empty>'}"
            )

        yield proc

        _stop_process(proc)


@pytest.fixture(scope="session")
def _pw():
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(app_process, _pw) -> Browser:
    """CDP 経由で WebView2 に接続するブラウザフィクスチャ。

    接続後、pywebview.api が利用可能になるまで待機する（初回のみ）。
    CI 環境では WebView2 の初期化に時間がかかるため 60 秒タイムアウトを設定。
    """
    b = _pw.chromium.connect_over_cdp(f"http://localhost:{REMOTE_DEBUG_PORT}")
    pg = b.contexts[0].pages[0]
    pg.wait_for_function(
        "typeof pywebview !== 'undefined' && typeof pywebview.api !== 'undefined'",
        timeout=60_000,
    )
    pg.wait_for_function(
        "window.__aviutlWhisperReady === true",
        timeout=60_000,
    )
    yield b
    # CDPセッションを切断（WebView2 プロセスは app_process が終了させる）
    b.close()


# ---------------------------------------------------------------------------
# 関数スコープ: ページ・モックセグメント
# ---------------------------------------------------------------------------

@pytest.fixture
def page(browser) -> Page:
    """pywebview API 準備完了済みのページを返すフィクスチャ。

    各テスト前にアプリ状態をリセットして前のテストの影響を排除する。
    browser フィクスチャがセッション開始時に pywebview の準備を保証済みなので
    ここでは追加待機不要。
    """
    context = browser.contexts[0]
    pg = context.pages[0]
    reset_app_state(pg)
    return pg


@pytest.fixture
def mock_segments(page: Page) -> Page:
    """モックセグメントを JS ステートに注入するフィクスチャ。

    セグメントテーブル・編集パネル・プレビューナビゲーションを更新済みの
    状態のページを返す。
    """
    data = json.loads((FIXTURES_DIR / "segments.json").read_text(encoding="utf-8"))
    segs_json = json.dumps(data["segments"])
    page.evaluate(f"""
        previewSegments = {segs_json};
        previewIndex = 0;
        isDirty = false;
        setTtsAvailability(true);
        document.getElementById('btn-save').disabled = false;
        document.getElementById('btn-save-project').disabled = false;
        document.getElementById('menu-save-project').disabled = false;
        document.getElementById('menu-save-project-as').disabled = false;
        document.getElementById('preview-placeholder').classList.add('hidden');
        renderSegmentTable();
        populateSegmentEditor();
        updatePreviewNav();
    """)
    return page
