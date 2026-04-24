"""UC-01: アプリ起動・初期 UI テスト"""

import pytest
from playwright.sync_api import Page

pytestmark = pytest.mark.e2e


def test_page_title(page: Page):
    """ページタイトルが正しい。"""
    assert "aviutl-whisper" in page.title()


def test_initial_elements_visible(page: Page):
    """主要な UI 要素が起動直後に表示されている。"""
    assert page.locator("#btn-select-file").is_visible()
    assert page.locator("#btn-start").is_visible()
    assert page.locator(".menu-bar").is_visible()


def test_transcription_buttons_disabled_initially(page: Page):
    """ファイル未選択時はセグメント操作・保存ボタンが無効。"""
    assert page.locator("#btn-save").is_disabled()
    assert page.locator("#btn-save-project").is_disabled()


def test_preview_placeholder_visible_initially(page: Page):
    """ファイル未選択時はプレビュープレースホルダーが表示されている。"""
    assert page.locator("#preview-placeholder").is_visible()


def test_segment_table_empty_state(page: Page):
    """起動直後はセグメントテーブルの空状態メッセージが表示されている。"""
    assert page.locator("#segment-table-empty").is_visible()
    assert page.locator("#segment-table-body").locator("tr").count() == 0


def test_preview_nav_initial_state(page: Page):
    """起動直後のプレビューナビゲーションは '- / -' を表示する。"""
    assert page.locator("#preview-seg-info").text_content() == "- / -"
    assert page.locator("#btn-prev-seg").is_disabled()
    assert page.locator("#btn-next-seg").is_disabled()
