"""UC-10: Undo / Redo テスト"""

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import get_segment_table_rows, mock_api_method

pytestmark = pytest.mark.e2e


def test_undo_button_disabled_initially(page: Page):
    """Undo ボタンは初期状態で無効。"""
    # メニューバーを開いてUndoボタンを確認
    page.locator("#menu-edit-entry .menu-entry-btn").click()
    page.locator("#menu-edit-dropdown").wait_for(state="visible")
    assert page.locator("#menu-undo").is_disabled()
    # メニューを閉じる
    page.keyboard.press("Escape")


def test_redo_button_disabled_initially(page: Page):
    """Redo ボタンは初期状態で無効。"""
    page.locator("#menu-edit-entry .menu-entry-btn").click()
    page.locator("#menu-edit-dropdown").wait_for(state="visible")
    assert page.locator("#menu-redo").is_disabled()
    page.keyboard.press("Escape")


def test_undo_enabled_after_segment_edit(mock_segments: Page):
    """セグメント編集後は Undo ボタンが有効になる。"""
    page = mock_segments
    mock_api_method(
        page,
        "update_segment",
        {
            "success": True,
            "segments": [
                {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": "変更後"},
                {"start": 3.8, "end": 7.2, "speaker": "Speaker 2", "text": "はじめまして。よろしくお願いします。"},
                {"start": 7.5, "end": 12.0, "speaker": "Speaker 1", "text": "今日はよい天気ですね。"},
            ],
        },
    )

    page.locator("#seg-edit-text").fill("変更後")
    page.locator("#btn-seg-apply").click()
    page.wait_for_timeout(300)

    page.locator("#menu-edit-entry .menu-entry-btn").click()
    page.locator("#menu-edit-dropdown").wait_for(state="visible")
    assert page.locator("#menu-undo").is_enabled()
    page.keyboard.press("Escape")


def test_undo_restores_previous_state(mock_segments: Page):
    """Undo を実行すると前のセグメントテキストに戻る。"""
    page = mock_segments
    original_text = "こんにちは、テストです。"

    mock_api_method(
        page,
        "update_segment",
        {
            "success": True,
            "segments": [
                {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": "変更後テキスト"},
                {"start": 3.8, "end": 7.2, "speaker": "Speaker 2", "text": "はじめまして。よろしくお願いします。"},
                {"start": 7.5, "end": 12.0, "speaker": "Speaker 1", "text": "今日はよい天気ですね。"},
            ],
        },
    )

    page.locator("#seg-edit-text").fill("変更後テキスト")
    page.locator("#btn-seg-apply").click()
    page.wait_for_timeout(300)

    # Undo 実行 (restore_segments をスタブ)
    mock_api_method(
        page,
        "restore_segments",
        {
            "success": True,
            "segments": [
                {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": original_text},
                {"start": 3.8, "end": 7.2, "speaker": "Speaker 2", "text": "はじめまして。よろしくお願いします。"},
                {"start": 7.5, "end": 12.0, "speaker": "Speaker 1", "text": "今日はよい天気ですね。"},
            ],
        },
    )

    page.keyboard.press("Control+z")
    page.wait_for_timeout(500)

    rows = get_segment_table_rows(page)
    assert rows[0]["text"] == original_text


def test_redo_restores_after_undo(mock_segments: Page):
    """Undo → Redo で変更後の状態が復元される。"""
    page = mock_segments
    edited_text = "Redo テスト用テキスト"

    mock_api_method(
        page,
        "update_segment",
        {
            "success": True,
            "segments": [
                {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": edited_text},
                {"start": 3.8, "end": 7.2, "speaker": "Speaker 2", "text": "はじめまして。よろしくお願いします。"},
                {"start": 7.5, "end": 12.0, "speaker": "Speaker 1", "text": "今日はよい天気ですね。"},
            ],
        },
    )
    page.locator("#seg-edit-text").fill(edited_text)
    page.locator("#btn-seg-apply").click()
    page.wait_for_timeout(300)

    # Undo
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
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)

    # Redo
    mock_api_method(
        page,
        "restore_segments",
        {
            "success": True,
            "segments": [
                {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": edited_text},
                {"start": 3.8, "end": 7.2, "speaker": "Speaker 2", "text": "はじめまして。よろしくお願いします。"},
                {"start": 7.5, "end": 12.0, "speaker": "Speaker 1", "text": "今日はよい天気ですね。"},
            ],
        },
    )
    page.keyboard.press("Control+y")
    page.wait_for_timeout(300)

    rows = get_segment_table_rows(page)
    assert rows[0]["text"] == edited_text


def test_clear_undo_history_disables_buttons(mock_segments: Page):
    """clearUndoHistory() 実行後は Undo / Redo ボタンが無効になる。"""
    page = mock_segments
    page.evaluate("clearUndoHistory()")

    page.locator("#menu-edit-entry .menu-entry-btn").click()
    page.locator("#menu-edit-dropdown").wait_for(state="visible")
    assert page.locator("#menu-undo").is_disabled()
    assert page.locator("#menu-redo").is_disabled()
    page.keyboard.press("Escape")
