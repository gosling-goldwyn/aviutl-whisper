"""E2E テスト共通ヘルパー関数"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Page

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def wait_for_pywebview(page: Page, timeout_ms: int = 15_000) -> None:
    """pywebview.api が利用可能になるまで待機する。"""
    page.wait_for_function(
        "typeof pywebview !== 'undefined' && typeof pywebview.api !== 'undefined'",
        timeout=timeout_ms,
    )


def inject_segments(page: Page, segments: list[dict] | None = None) -> None:
    """previewSegments を差し替えてセグメント関連 UI を再描画する。

    segments が None の場合は fixtures/segments.json のデータを使用する。
    """
    if segments is None:
        data = json.loads((FIXTURES_DIR / "segments.json").read_text(encoding="utf-8"))
        segments = data["segments"]

    segs_json = json.dumps(segments)
    page.evaluate(f"""
        previewSegments = {segs_json};
        previewIndex = 0;
        document.getElementById('preview-placeholder').classList.add('hidden');
        renderSegmentTable();
        populateSegmentEditor();
        updatePreviewNav();
    """)


def mock_api_method(page: Page, method: str, return_value: Any) -> None:
    """pywebview.api のメソッドを固定値を返すスタブに差し替える。

    Args:
        page: Playwright ページ
        method: メソッド名（例: "select_file"）
        return_value: JSON シリアライズ可能な戻り値
    """
    rv_json = json.dumps(return_value)
    page.evaluate(f"""
        pywebview.api.{method} = () => Promise.resolve({rv_json});
    """)


def reset_app_state(page: Page) -> None:
    """アプリの編集状態をリセットし、空の初期状態に戻す。

    各テスト前に page フィクスチャから呼び出され、前のテストの状態を
    クリアする。render_preview_frame もスタブ化してブロッキングを防ぐ。
    """
    page.evaluate("""
        document.getElementById('transcription-modal').classList.add('hidden');
        document.getElementById('save-confirm-modal').classList.add('hidden');
        document.querySelectorAll('.menu-entry').forEach(e => e.classList.remove('open'));
        previewSegments = [];
        previewIndex = 0;
        isDirty = false;
        selectedFile = null;
        isProcessing = false;
        clearUndoHistory();
        document.getElementById('segment-table-body').innerHTML = '';
        document.getElementById('segment-table-empty').classList.remove('hidden');
        hide(document.getElementById('seg-editor'));
        document.getElementById('preview-placeholder').classList.remove('hidden');
        document.getElementById('preview-seg-info').textContent = '- / -';
        document.getElementById('btn-prev-seg').disabled = true;
        document.getElementById('btn-next-seg').disabled = true;
        document.getElementById('btn-start').disabled = true;
        document.getElementById('file-name').textContent = '未選択';
        document.getElementById('file-info').classList.add('hidden');
        document.getElementById('progress-area').classList.add('hidden');
        document.getElementById('btn-save').disabled = true;
        document.getElementById('btn-save-project').disabled = true;
        document.getElementById('menu-save-project').disabled = true;
        document.getElementById('menu-save-project-as').disabled = true;
        hide(document.getElementById('speaker-mapping-section'));
        pywebview.api.render_preview_frame = () => Promise.resolve({
            success: true,
            data_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII='
        });
    """)


def get_segment_table_rows(page: Page) -> list[dict[str, str]]:
    """セグメントテーブルの全行を辞書リストとして返す。"""
    return page.evaluate("""
        () => {
            const rows = document.querySelectorAll('#segment-table-body tr');
            return Array.from(rows).map(tr => {
                const cells = tr.querySelectorAll('td');
                return {
                    time:    cells[0]?.textContent?.trim() || '',
                    speaker: cells[1]?.textContent?.trim() || '',
                    text:    cells[2]?.textContent?.trim() || '',
                    active:  tr.classList.contains('active'),
                };
            });
        }
    """)


def get_preview_nav_text(page: Page) -> str:
    """プレビューナビゲーション（例: "1 / 3"）のテキストを返す。"""
    return page.locator("#preview-seg-info").text_content() or ""


def dismiss_modal(page: Page, modal_id: str, button_selector: str) -> None:
    """指定モーダルのボタンをクリックして閉じる。"""
    page.locator(f"#{modal_id}").wait_for(state="visible")
    page.locator(button_selector).click()
    page.locator(f"#{modal_id}").wait_for(state="hidden")


def open_menu_item(page: Page, menu_btn_id: str, item_id: str) -> None:
    """メニューボタンを開いてアイテムをクリックする。"""
    page.locator(f"#{menu_btn_id}").click()
    page.locator(f"#{item_id}").wait_for(state="visible")
    page.locator(f"#{item_id}").click()
