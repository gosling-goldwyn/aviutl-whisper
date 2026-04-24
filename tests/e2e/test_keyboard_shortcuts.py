"""UC-11: キーボードショートカットテスト"""

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import get_preview_nav_text, mock_api_method

pytestmark = pytest.mark.e2e


def test_ctrl_z_triggers_undo(mock_segments: Page):
    """Ctrl+Z で Undo が実行される。"""
    page = mock_segments
    mock_api_method(
        page,
        "update_segment",
        {
            "success": True,
            "segments": [
                {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": "ショートカットテスト"},
                {"start": 3.8, "end": 7.2, "speaker": "Speaker 2", "text": "はじめまして。"},
                {"start": 7.5, "end": 12.0, "speaker": "Speaker 1", "text": "よい天気。"},
            ],
        },
    )
    page.locator("#seg-edit-text").fill("ショートカットテスト")
    page.locator("#btn-seg-apply").click()
    page.wait_for_timeout(300)

    # undo stack が空でないことを確認
    undo_len = page.evaluate("undoStack.length")
    assert undo_len > 0

    mock_api_method(
        page,
        "restore_segments",
        {
            "success": True,
            "segments": [
                {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": "こんにちは、テストです。"},
                {"start": 3.8, "end": 7.2, "speaker": "Speaker 2", "text": "はじめまして。よろしくお願いします。"},
                {"start": 7.5, "end": 12.0, "speaker": "Speaker 1", "text": "今日はよい天気ですね。"},
            ],
        },
    )

    # textarea にフォーカスがある場合はブラウザのデフォルトが優先されるので body にフォーカスを移す
    page.evaluate("document.activeElement?.blur()")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)

    # Undo 後は undoStack が減る
    new_undo_len = page.evaluate("undoStack.length")
    assert new_undo_len < undo_len


def test_ctrl_y_triggers_redo(mock_segments: Page):
    """Ctrl+Y で Redo が実行される。"""
    page = mock_segments

    # まず何かを Undo して Redo できる状態にする
    page.evaluate("""
        const snap = captureSnapshot();
        undoStack.push(snap);
        updateUndoRedoUI();
        // undo を手動実行して redo stack を作る
        redoStack.push(snap);
        updateUndoRedoUI();
    """)

    redo_len_before = page.evaluate("redoStack.length")
    assert redo_len_before > 0

    mock_api_method(
        page,
        "restore_segments",
        {
            "success": True,
            "segments": [
                {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": "こんにちは、テストです。"},
                {"start": 3.8, "end": 7.2, "speaker": "Speaker 2", "text": "はじめまして。よろしくお願いします。"},
                {"start": 7.5, "end": 12.0, "speaker": "Speaker 1", "text": "今日はよい天気ですね。"},
            ],
        },
    )

    page.evaluate("document.activeElement?.blur()")
    page.keyboard.press("Control+y")
    page.wait_for_timeout(300)

    redo_len_after = page.evaluate("redoStack.length")
    assert redo_len_after < redo_len_before


def test_arrow_keys_navigate_segments(mock_segments: Page):
    """左右矢印キーでセグメントナビゲーションができる。"""
    page = mock_segments
    # ページ本体（textarea 以外）にフォーカス
    page.evaluate("document.activeElement?.blur()")
    page.locator("body").click()

    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(200)
    assert get_preview_nav_text(page) == "2 / 3"

    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(200)
    assert get_preview_nav_text(page) == "1 / 3"


def test_ctrl_s_saves_project(mock_segments: Page):
    """Ctrl+S でプロジェクト保存が実行される。"""
    page = mock_segments
    page.evaluate("""
        window.__ctrlSSaved = false;
        pywebview.api.save_project = (data) => {
            window.__ctrlSSaved = true;
            return Promise.resolve({ success: true, path: '/tmp/shortcut.awp' });
        };
    """)

    page.evaluate("document.activeElement?.blur()")
    page.keyboard.press("Control+s")
    page.wait_for_function("window.__ctrlSSaved === true", timeout=3000)
    assert page.evaluate("window.__ctrlSSaved") is True
