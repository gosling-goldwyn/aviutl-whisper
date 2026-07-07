"""ElevenLabs TTS modal model-dependent UI tests."""

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import inject_segments


pytestmark = pytest.mark.e2e


def test_eleven_v3_switches_to_chunk_mode(page: Page):
    page.evaluate("""
        renderTtsModelOptions(
            [{ model_id: 'eleven_v3', name: 'Eleven v3' }],
            'eleven_v3'
        );
        document.getElementById('tts-modal').classList.remove('hidden');
    """)
    assert page.locator("#tts-generation-mode").input_value() == "chunk_v3"
    assert page.locator("#tts-v3-warning").is_visible()
    assert "language text normalization" in page.locator(
        "#tts-v3-warning"
    ).inner_text()
    assert not page.locator("#tts-multilingual-info").is_visible()
    assert page.locator("#tts-chunk-settings").is_visible()
    assert (
        page.locator("#btn-generate-tts-selected").inner_text()
        == "選択チャンクTTS生成"
    )


def test_non_v3_uses_per_segment_mode(page: Page):
    page.evaluate("""
        renderTtsModelOptions(
            [{
                model_id: 'eleven_multilingual_v2',
                name: 'Multilingual v2'
            }],
            'eleven_multilingual_v2'
        );
        document.getElementById('tts-modal').classList.remove('hidden');
    """)
    assert (
        page.locator("#tts-generation-mode").input_value()
        == "per_segment_context"
    )
    assert not page.locator("#tts-v3-warning").is_visible()
    assert page.locator("#tts-multilingual-info").is_visible()
    assert "previous_text / next_text は使用できます" in page.locator(
        "#tts-multilingual-info"
    ).inner_text()
    assert not page.locator("#tts-chunk-settings").is_visible()


def test_tts_entries_disabled_without_segments(page: Page):
    assert page.locator("#btn-open-tts").is_disabled()
    page.locator("#menu-tools-entry .menu-entry-btn").click()
    assert page.locator("#menu-open-tts").is_disabled()
    page.keyboard.press("Control+Shift+E")
    assert not page.locator("#tts-modal").is_visible()


def test_tts_opens_from_tools_menu_and_shortcut(page: Page):
    inject_segments(page)
    assert page.locator("#btn-open-tts").is_enabled()
    page.locator("#btn-open-tts").click()
    page.locator("#tts-modal").wait_for(state="visible")
    page.locator("#btn-close-tts").click()
    page.locator("#tts-modal").wait_for(state="hidden")

    page.locator("#menu-tools-entry .menu-entry-btn").click()
    assert page.locator("#menu-open-tts").is_enabled()
    page.locator("#menu-open-tts").click()
    page.locator("#tts-modal").wait_for(state="visible")
    page.locator("#btn-close-tts").click()
    page.locator("#tts-modal").wait_for(state="hidden")

    page.keyboard.press("Control+Shift+E")
    page.locator("#tts-modal").wait_for(state="visible")


def test_selected_segment_tts_panel_status_and_actions(page: Page):
    page.evaluate("""
        window.__ttsPlayedIndex = null;
        window.__ttsGeneratedIndex = null;
        pywebview.api.get_tts_status = () => Promise.resolve({
            success: true,
            segments: [{
                index: 0,
                speaker: 'Speaker 1',
                text: 'テスト',
                voice_id: 'voice-one',
                audio_path: 'C:/tmp/0001.mp3',
                status: 'generated',
                error: ''
            }]
        });
        pywebview.api.play_tts_segment = (index) => {
            window.__ttsPlayedIndex = index;
            return Promise.resolve({ success: true });
        };
        pywebview.api.generate_tts_for_segment = (index) => {
            window.__ttsGeneratedIndex = index;
            return Promise.resolve({
                success: true,
                index,
                status: 'generated',
                audio_path: 'C:/tmp/0001.mp3'
            });
        };
        pywebview.api.get_tts_progress = () => Promise.resolve({
            progress: 1,
            message: '完了'
        });
    """)
    inject_segments(page, [{
        "start": 0,
        "end": 1,
        "speaker": "Speaker 1",
        "text": "テスト",
    }])
    page.evaluate("refreshSelectedTtsStatus()")
    page.locator("#seg-tts-status").wait_for(state="visible")
    assert page.locator("#seg-tts-status").inner_text() == "生成済み"
    assert page.locator("#btn-seg-tts-play").is_enabled()

    page.locator("#btn-seg-tts-play").click()
    page.wait_for_function("window.__ttsPlayedIndex === 0")
    page.locator("#btn-seg-tts-regenerate").click()
    page.wait_for_function("window.__ttsGeneratedIndex === 0")
