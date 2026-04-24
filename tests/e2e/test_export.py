"""UC-09: EXO エクスポートテスト"""

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import mock_api_method

pytestmark = pytest.mark.e2e


def test_save_button_disabled_initially(page: Page):
    """起動直後は exo 保存ボタンが無効。"""
    assert page.locator("#btn-save").is_disabled()


def test_save_button_enabled_after_segments(mock_segments: Page):
    """セグメント注入後は exo 保存ボタンが有効。"""
    assert mock_segments.locator("#btn-save").is_enabled()


def test_save_calls_api_with_settings(mock_segments: Page):
    """exo 保存ボタンクリック時に save_result API が呼ばれる。"""
    page = mock_segments
    page.evaluate("""
        window.__saveArgs = null;
        pywebview.api.save_result = (format, settings, mapping) => {
            window.__saveArgs = { format, settings, mapping };
            return Promise.resolve({ success: true, path: '/tmp/output.exo' });
        };
    """)

    page.locator("#btn-save").click()
    page.wait_for_function("window.__saveArgs !== null")

    args = page.evaluate("window.__saveArgs")
    assert args["format"] == "exo"
    assert args["settings"] is not None


def test_save_includes_font_size(mock_segments: Page):
    """保存時に現在のフォントサイズが設定に含まれる。"""
    page = mock_segments
    page.locator("#exo-font-size").fill("48")

    page.evaluate("""
        window.__capturedSettings = null;
        pywebview.api.save_result = (format, settings, mapping) => {
            window.__capturedSettings = settings;
            return Promise.resolve({ success: true, path: '/tmp/output.exo' });
        };
    """)

    page.locator("#btn-save").click()
    page.wait_for_function("window.__capturedSettings !== null")

    settings = page.evaluate("window.__capturedSettings")
    assert settings["font_size"] == 48

    # 後片付け
    page.locator("#exo-font-size").fill("34")


def test_save_result_cancel_no_error(mock_segments: Page):
    """保存キャンセル時（error: 'キャンセルされました'）はエラーアラートが表示されない。"""
    page = mock_segments
    page.evaluate("""
        window.__alertCalled = false;
        const origAlert = window.alert;
        window.alert = (msg) => { window.__alertCalled = true; origAlert(msg); };
        pywebview.api.save_result = () =>
            Promise.resolve({ success: false, error: 'キャンセルされました' });
    """)

    page.locator("#btn-save").click()
    page.wait_for_timeout(500)
    assert page.evaluate("window.__alertCalled") is False
