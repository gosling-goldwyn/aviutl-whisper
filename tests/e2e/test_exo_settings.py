"""UC-05: EXO 詳細設定テスト"""

import pytest
from playwright.sync_api import Page

pytestmark = pytest.mark.e2e


def test_default_font_size(page: Page):
    """デフォルトのフォントサイズが 34 であること。"""
    assert page.locator("#exo-font-size").input_value() == "34"


def test_default_align_center(page: Page):
    """デフォルトの寄せ方向が '中央' (value=4) であること。"""
    assert page.locator("#exo-align").input_value() == "4"


def test_soft_edge_checked_by_default(page: Page):
    """デフォルトで縁取りチェックボックスがオンであること。"""
    assert page.locator("#exo-soft-edge").is_checked()


def test_bold_unchecked_by_default(page: Page):
    """デフォルトで太字チェックボックスがオフであること。"""
    assert not page.locator("#exo-bold").is_checked()


def test_italic_unchecked_by_default(page: Page):
    """デフォルトで斜体チェックボックスがオフであること。"""
    assert not page.locator("#exo-italic").is_checked()


def test_font_size_change_accepted(page: Page):
    """フォントサイズを変更できる。"""
    page.locator("#exo-font-size").fill("48")
    assert page.locator("#exo-font-size").input_value() == "48"
    # 後片付け
    page.locator("#exo-font-size").fill("34")


def test_bold_toggle(page: Page):
    """太字チェックボックスをトグルできる。"""
    page.locator("#exo-bold").check()
    assert page.locator("#exo-bold").is_checked()
    page.locator("#exo-bold").uncheck()
    assert not page.locator("#exo-bold").is_checked()


def test_align_select_options(page: Page):
    """寄せ方向セレクトに 9 つのオプションがある。"""
    options = page.locator("#exo-align option").all()
    assert len(options) == 9


def test_max_chars_default(page: Page):
    """デフォルトの 1 行最大文字数が 20 であること。"""
    assert page.locator("#exo-max-chars").input_value() == "20"
