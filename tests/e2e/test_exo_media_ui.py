"""音声・BGM・クロージング設定の操作と保存・復元。"""

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def restore_media_state(page: Page):
    page.evaluate("""() => {
        window.originalMusicPicker = pywebview.api.select_music_files;
        window.originalImagePicker = pywebview.api.select_image_file;
        window.originalSettingsExporter = pywebview.api.export_settings;
        applyExoMediaToUI();
        clearUndoHistory();
    }""")
    yield
    page.evaluate("""async () => {
        pywebview.api.select_music_files = window.originalMusicPicker;
        pywebview.api.select_image_file = window.originalImagePicker;
        pywebview.api.export_settings = window.originalSettingsExporter;
        clearTimeout(saveTimer);
        applyExoMediaToUI();
        await pywebview.api.save_settings(collectAllSettings());
        clearUndoHistory();
    }""")


def test_media_selection_and_settings_roundtrip(page: Page):
    page.evaluate("""() => {
        const selections = [
            ['C:/music/start.wav'], ['C:/music/end.wav'],
            ['C:/music/one.mp3', 'C:/music/two.mp3'], ['C:/music/jingle.wav'],
        ];
        pywebview.api.select_music_files = () => Promise.resolve(selections.shift());
        pywebview.api.select_image_file = () => Promise.resolve('C:/images/closing.png');
    }""")
    page.locator('[data-media-select="bgm_start_file"]').click()
    page.locator('[data-media-select="bgm_end_file"]').click()
    page.locator("#btn-bgm-add").click()
    expect(page.locator("#bgm-file-list .bg-image-row")).to_have_count(2)
    page.locator('[data-media-select="closing_jingle"]').click()
    page.locator('[data-media-select="closing_image"]').click()
    page.locator("#exo-audio-start").fill("2.5")
    page.locator("#exo-bgm-volume").fill("0")
    page.locator("#exo-outro-seconds").fill("3")
    page.locator("#exo-closing-seconds").fill("12")
    saved = page.evaluate("""async () => {
        await pywebview.api.save_settings(collectAllSettings());
        const saved = await pywebview.api.load_settings();
        applyExoMediaToUI();
        applySettingsToUi(saved);
        return collectExoSettings();
    }""")
    assert saved["audio_start_seconds"] == 2.5
    assert saved["bgm_volume"] == 0
    assert saved["outro_seconds"] == 3
    assert saved["closing_seconds"] == 12
    assert saved["bgm_files"] == ["C:/music/one.mp3", "C:/music/two.mp3"]
    assert saved["bgm_start_file"] == "C:/music/start.wav"
    assert saved["bgm_end_file"] == "C:/music/end.wav"
    assert saved["closing_jingle"] == "C:/music/jingle.wav"
    assert saved["closing_image"] == "C:/images/closing.png"
    expect(page.locator('[data-media-name="closing_image"]')).to_have_text("closing.png")
    expect(page.locator("#bgm-file-list .bg-image-row")).to_have_count(2)


def test_remove_and_undo_restore_tracks_and_optional_ending(page: Page):
    page.evaluate("""() => {
        applyExoMediaToUI({bgm_files: ['one.wav', 'two.wav'], bgm_end_file: 'end.wav'});
    }""")
    page.get_by_role("button", name="one.wavを削除").click()
    expect(page.locator("#bgm-file-list .bg-image-row")).to_have_count(1)
    page.evaluate("undo()")
    expect(page.locator("#bgm-file-list .bg-image-row")).to_have_count(2)
    page.locator('[data-media-clear="bgm_end_file"]').click()
    expect(page.locator('[data-media-name="bgm_end_file"]')).to_have_text("未選択")
    page.evaluate("undo()")
    expect(page.locator('[data-media-name="bgm_end_file"]')).to_have_text("end.wav")


def test_export_settings_includes_changes_before_debounced_save(page: Page):
    page.evaluate("""() => {
        window.exportedMediaSettings = null;
        pywebview.api.export_settings = async () => {
            window.exportedMediaSettings = (await pywebview.api.load_settings()).exo;
            return {success: true};
        };
    }""")
    page.locator("#exo-bgm-volume").fill("37")
    page.evaluate("exportSettings()")
    page.wait_for_function("window.exportedMediaSettings !== null")
    assert page.evaluate("window.exportedMediaSettings.bgm_volume") == 37
