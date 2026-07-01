"""4ペインレイアウトのE2Eテスト"""

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def _reset_layout(page: Page) -> None:
    page.evaluate("""
        localStorage.removeItem('aviutlWhisper.leftSidebarCollapsed');
        localStorage.removeItem('aviutlWhisper.rightPaneWidth');
        localStorage.removeItem('aviutlWhisper.segmentListHeight');
        setSettingsPaneCollapsed(false);
        document.getElementById('app-layout').style.removeProperty('--right-pane-w');
        document.getElementById('app-layout').style.removeProperty('--segment-list-h');
    """)


def _css_px(page: Page, name: str) -> float:
    return page.evaluate(
        """name => parseFloat(
            getComputedStyle(document.getElementById('app-layout')).getPropertyValue(name)
        )""",
        name,
    )


def test_settings_sidebar_toggle(page: Page):
    _reset_layout(page)
    toggle = page.locator("#btn-toggle-settings-pane")

    assert not page.locator("#app-layout").evaluate("el => el.classList.contains('settings-collapsed')")
    toggle.click()
    assert page.locator("#app-layout").evaluate("el => el.classList.contains('settings-collapsed')")
    expect(toggle).to_have_attribute("aria-expanded", "false")
    toggle.click()
    assert not page.locator("#app-layout").evaluate("el => el.classList.contains('settings-collapsed')")
    expect(toggle).to_have_attribute("aria-expanded", "true")


def test_right_editor_pane_resizes_by_drag(mock_segments: Page):
    page = mock_segments
    _reset_layout(page)
    splitter = page.locator("#right-splitter").bounding_box()
    assert splitter is not None

    before = page.locator("#segment-editor-pane").bounding_box()
    assert before is not None

    page.mouse.move(splitter["x"] + splitter["width"] / 2, splitter["y"] + 40)
    page.mouse.down()
    page.mouse.move(splitter["x"] + 80, splitter["y"] + 40)
    page.mouse.up()
    page.wait_for_timeout(250)

    after = page.locator("#segment-editor-pane").bounding_box()
    assert after is not None
    assert after["width"] < before["width"] - 40


def test_segment_list_height_resizes_by_drag(mock_segments: Page):
    page = mock_segments
    _reset_layout(page)
    splitter = page.locator("#center-splitter").bounding_box()
    assert splitter is not None

    before = _css_px(page, "--segment-list-h")

    page.mouse.move(splitter["x"] + 40, splitter["y"] + splitter["height"] / 2)
    page.mouse.down()
    page.mouse.move(splitter["x"] + 40, splitter["y"] - 70)
    page.mouse.up()

    after = _css_px(page, "--segment-list-h")
    assert after > before + 40


def test_four_panes_keep_segment_selection_in_sync(mock_segments: Page):
    page = mock_segments
    _reset_layout(page)

    expect(page.locator("#settings-pane")).to_be_visible()
    expect(page.locator("#preview-section")).to_be_visible()
    expect(page.locator("#segment-table-section")).to_be_visible()
    expect(page.locator("#segment-editor-pane")).to_be_visible()

    page.locator("#segment-table-body tr").nth(1).click()

    expect(page.locator("#preview-seg-info")).to_have_text("2 / 3")
    expect(page.locator("#seg-edit-text")).to_have_value("はじめまして。よろしくお願いします。")
