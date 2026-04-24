"""UC-02: 音声ファイル選択テスト"""

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import mock_api_method

pytestmark = pytest.mark.e2e


def test_file_selected_shows_filename(page: Page):
    """ファイルが選択されると選択ファイル名が UI に表示される。"""
    mock_api_method(
        page,
        "select_file",
        {"path": "/tmp/test_audio.m4a", "name": "test_audio.m4a", "extension": ".m4a", "size": 1024},
    )
    page.locator("#btn-select-file").click()
    page.wait_for_function(
        "document.getElementById('file-name').textContent.includes('test_audio.m4a')"
    )
    assert "test_audio.m4a" in page.locator("#file-name").text_content()


def test_file_select_cancel_keeps_previous_state(page: Page):
    """キャンセルした場合は選択ファイル名が変更されない。"""
    # まず既存ファイルを設定
    mock_api_method(
        page,
        "select_file",
        {"path": "/tmp/original.m4a", "name": "original.m4a", "extension": ".m4a", "size": 2048},
    )
    page.locator("#btn-select-file").click()
    page.wait_for_function(
        "document.getElementById('file-name').textContent.includes('original.m4a')"
    )

    # キャンセル（null を返すと JS 側の if (result) が偽になり UI 更新なし）
    mock_api_method(page, "select_file", None)
    page.locator("#btn-select-file").click()
    page.wait_for_timeout(500)

    assert "original.m4a" in page.locator("#file-name").text_content()


def test_start_button_enabled_after_file_select(page: Page):
    """ファイル選択後に文字起こし開始ボタンが有効になる。"""
    mock_api_method(
        page,
        "select_file",
        {"path": "/tmp/sample.wav", "name": "sample.wav", "extension": ".wav", "size": 512},
    )
    page.locator("#btn-select-file").click()
    page.wait_for_function(
        "document.getElementById('file-name').textContent.includes('sample.wav')"
    )
    assert page.locator("#btn-start").is_enabled()
