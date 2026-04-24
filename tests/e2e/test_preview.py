"""UC-06: シーンプレビューテスト"""

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import get_preview_nav_text

pytestmark = pytest.mark.e2e


def test_placeholder_hidden_after_inject(mock_segments: Page):
    """セグメント注入後はプレビュープレースホルダーが非表示。"""
    assert not mock_segments.locator("#preview-placeholder").is_visible()


def test_preview_nav_initial_count(mock_segments: Page):
    """プレビューナビは '1 / 3' を表示する。"""
    assert get_preview_nav_text(mock_segments) == "1 / 3"


def test_prev_button_disabled_on_first(mock_segments: Page):
    """先頭セグメントでは '前へ' ボタンが無効。"""
    assert mock_segments.locator("#btn-prev-seg").is_disabled()


def test_next_button_enabled_on_first(mock_segments: Page):
    """先頭セグメントでは '次へ' ボタンが有効。"""
    assert mock_segments.locator("#btn-next-seg").is_enabled()


def test_next_advances_nav(mock_segments: Page):
    """'次へ' ボタンをクリックするとナビゲーションが '2 / 3' に更新される。"""
    mock_segments.locator("#btn-next-seg").click()
    assert get_preview_nav_text(mock_segments) == "2 / 3"


def test_prev_goes_back(mock_segments: Page):
    """'次へ' → '前へ' で元の '1 / 3' に戻る。"""
    mock_segments.locator("#btn-next-seg").click()
    mock_segments.locator("#btn-prev-seg").click()
    assert get_preview_nav_text(mock_segments) == "1 / 3"


def test_last_segment_disables_next(mock_segments: Page):
    """末尾セグメントでは '次へ' ボタンが無効。"""
    mock_segments.locator("#btn-next-seg").click()
    mock_segments.locator("#btn-next-seg").click()
    assert mock_segments.locator("#btn-next-seg").is_disabled()
    assert mock_segments.locator("#btn-prev-seg").is_enabled()


def test_nav_syncs_with_table_click(mock_segments: Page):
    """セグメントテーブルの行クリックでプレビューナビが同期する。"""
    mock_segments.locator("#segment-table-body tr").nth(2).click()
    assert get_preview_nav_text(mock_segments) == "3 / 3"
