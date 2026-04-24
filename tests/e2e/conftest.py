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


# ---------------------------------------------------------------------------
# セッションスコープ: アプリプロセス・ブラウザ接続
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app_process(tmp_path_factory):
    """WebView2 リモートデバッグポート付きでアプリをサブプロセス起動するフィクスチャ。

    - ``AVIUTL_WHISPER_HIDDEN=1`` で不可視ウィンドウ起動（ヘッドレス相当）
    - ``LOCALAPPDATA`` を一時ディレクトリに差し替えて settings.json を分離
    """
    tmp_settings = tmp_path_factory.mktemp("settings")

    env = os.environ.copy()
    env["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = (
        f"--remote-debugging-port={REMOTE_DEBUG_PORT}"
    )
    env["AVIUTL_WHISPER_HIDDEN"] = "1"
    env["LOCALAPPDATA"] = str(tmp_settings)

    python = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    proc = subprocess.Popen(
        [python, str(PROJECT_ROOT / "main.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    # CDP エンドポイントが利用可能になるまでポーリング
    deadline = time.time() + APP_LAUNCH_TIMEOUT
    connected = False
    while time.time() < deadline:
        try:
            urllib.request.urlopen(
                f"http://localhost:{REMOTE_DEBUG_PORT}/json/version",
                timeout=1,
            )
            connected = True
            break
        except Exception:
            time.sleep(0.5)

    if not connected:
        proc.terminate()
        pytest.fail(
            f"App did not expose CDP on port {REMOTE_DEBUG_PORT} "
            f"within {APP_LAUNCH_TIMEOUT}s"
        )

    # WebView2 の初期化バッファ
    time.sleep(2)

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def _pw():
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(app_process, _pw) -> Browser:
    """CDP 経由で WebView2 に接続するブラウザフィクスチャ。"""
    b = _pw.chromium.connect_over_cdp(f"http://localhost:{REMOTE_DEBUG_PORT}")
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
    """
    context = browser.contexts[0]
    pg = context.pages[0]
    pg.wait_for_function(
        "typeof pywebview !== 'undefined' && typeof pywebview.api !== 'undefined'",
        timeout=15_000,
    )
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
