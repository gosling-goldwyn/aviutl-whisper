"""UC-07: 話者カラー・立ち絵設定テスト"""

import pytest
from playwright.sync_api import Page

pytestmark = pytest.mark.e2e


def test_speaker_colors_section_visible(page: Page):
    """話者ごとの色設定セクションが表示されている。"""
    assert page.locator("#speaker-colors-section").is_visible()


def test_speaker_tachie_section_visible(page: Page):
    """話者ごとの立ち絵設定セクションが表示されている。"""
    assert page.locator("#speaker-tachie-section").is_visible()


def test_bg_section_visible(page: Page):
    """背景画像セクションが表示されている。"""
    assert page.locator("#bg-section").is_visible()


def test_bg_image_initial_state(page: Page):
    """背景画像の初期状態は '未選択' と表示される。"""
    assert page.locator("#bg-image-name").text_content() == "未選択"


def test_speaker_mapping_hidden_initially(page: Page):
    """話者マッピングセクションは初期状態では非表示。"""
    assert not page.locator("#speaker-mapping-section").is_visible()


def test_speaker_mapping_shown_after_inject(page: Page):
    """showResult で 2 人の話者を渡すと話者マッピングセクションが表示される。"""
    page.evaluate("""
        showResult({
            success: true,
            num_segments: 3,
            num_speakers: 2,
            language: 'ja',
            speakers: [
                { name: 'Speaker 1', sample_text: 'こんにちは', segment_count: 2, first_start: 0, first_end: 3.5 },
                { name: 'Speaker 2', sample_text: 'よろしく', segment_count: 1, first_start: 3.8, first_end: 7.2 }
            ],
            text: 'こんにちはよろしく'
        });
    """)
    page.locator("#speaker-mapping-section").wait_for(state="visible")
    assert page.locator("#speaker-mapping-section").is_visible()


def test_swap_speakers_button_visible_for_two_speakers(page: Page):
    """話者 2 人のとき '話者を入れ替え' ボタンが表示される。"""
    # 前のテストで speaker-mapping-section が表示されていることを前提とする
    # または個別に showResult を呼ぶ
    page.evaluate("""
        renderSpeakerMapping([
            { name: 'Speaker 1', sample_text: 'テスト', segment_count: 1, first_start: 0, first_end: 1 },
            { name: 'Speaker 2', sample_text: 'サンプル', segment_count: 1, first_start: 1, first_end: 2 }
        ]);
        show(document.getElementById('speaker-mapping-section'));
    """)
    assert page.locator("#btn-swap-speakers").is_visible()
