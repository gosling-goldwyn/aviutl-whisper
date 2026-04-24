"""UC-08: プロジェクト保存・読み込み・未保存確認テスト"""

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import mock_api_method

pytestmark = pytest.mark.e2e


def test_save_project_button_disabled_initially(page: Page):
    """起動直後はプロジェクト保存ボタンが無効。"""
    assert page.locator("#btn-save-project").is_disabled()


def test_save_project_enabled_after_segments(mock_segments: Page):
    """セグメント注入後はプロジェクト保存ボタンが有効。"""
    assert mock_segments.locator("#btn-save-project").is_enabled()


def test_save_project_calls_api(mock_segments: Page):
    """プロジェクト保存ボタンが save_project API を呼び出す。"""
    page = mock_segments
    called = page.evaluate("window.__saveProjectCalled = false; undefined")
    page.evaluate("""
        const orig = pywebview.api.save_project;
        pywebview.api.save_project = (data) => {
            window.__saveProjectCalled = true;
            return Promise.resolve({ success: true, path: '/tmp/test.awp' });
        };
    """)

    page.locator("#btn-save-project").click()
    page.wait_for_function("window.__saveProjectCalled === true")
    assert page.evaluate("window.__saveProjectCalled") is True


def test_unsaved_confirm_dialog_on_open_with_dirty(mock_segments: Page):
    """未保存の変更があるときにプロジェクトを開こうとすると確認ダイアログが表示される。"""
    page = mock_segments
    # isDirty にする
    page.evaluate("markDirty()")

    # load_project は呼ばれないようにスタブ（キャンセル）
    page.evaluate("""
        pywebview.api.load_project = () => Promise.resolve({ success: false, error: 'キャンセルされました' });
        pywebview.api.save_project = () => Promise.resolve({ success: false, error: 'キャンセルされました' });
    """)

    page.locator("#btn-load-project").click()
    page.locator("#save-confirm-modal").wait_for(state="visible")
    assert page.locator("#save-confirm-modal").is_visible()

    # キャンセルで閉じる
    page.locator("#btn-confirm-cancel").click()
    page.locator("#save-confirm-modal").wait_for(state="hidden")


def test_confirm_dialog_no_discards_changes(mock_segments: Page):
    """確認ダイアログで 'いいえ' を選ぶと保存せずにプロジェクトを開く処理が走る。"""
    page = mock_segments
    page.evaluate("markDirty()")

    page.evaluate("""
        window.__loadCalled = false;
        pywebview.api.save_project = () => Promise.resolve({ success: false, error: 'キャンセルされました' });
        pywebview.api.load_project = () => {
            window.__loadCalled = true;
            return Promise.resolve({ success: false, error: 'キャンセルされました' });
        };
    """)

    page.locator("#btn-load-project").click()
    page.locator("#save-confirm-modal").wait_for(state="visible")
    page.locator("#btn-confirm-no").click()

    page.wait_for_function("window.__loadCalled === true")
    assert page.evaluate("window.__loadCalled") is True


def test_load_project_updates_ui(page: Page):
    """プロジェクト読み込み成功後にセグメントテーブルが更新される。"""
    mock_api_method(
        page,
        "load_project",
        {
            "success": True,
            "source_file": "/tmp/audio.m4a",
            "num_segments": 2,
            "num_speakers": 2,
            "language": "ja",
            "speakers": [
                {"name": "Speaker 1", "sample_text": "テスト", "segment_count": 1, "first_start": 0, "first_end": 1},
                {"name": "Speaker 2", "sample_text": "サンプル", "segment_count": 1, "first_start": 1, "first_end": 2},
            ],
            "exo_settings": {},
            "preview_index": 0,
        },
    )
    mock_api_method(
        page,
        "get_preview_segments",
        {
            "success": True,
            "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "Speaker 1", "text": "テスト"},
                {"start": 1.0, "end": 2.0, "speaker": "Speaker 2", "text": "サンプル"},
            ],
        },
    )

    page.locator("#btn-load-project").click()
    page.wait_for_function(
        "document.querySelectorAll('#segment-table-body tr').length >= 2"
    )
    assert page.locator("#segment-table-body tr").count() == 2
