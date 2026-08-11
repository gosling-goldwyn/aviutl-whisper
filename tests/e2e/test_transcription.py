"""UC-03: 文字起こし設定・実行・キャンセルテスト"""

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import mock_api_method

pytestmark = pytest.mark.e2e


def test_settings_modal_opens(page: Page):
    """設定（詳細設定）モーダルが開ける。"""
    page.locator("#btn-open-settings").click()
    page.locator("#transcription-modal").wait_for(state="visible")
    assert page.locator("#transcription-modal").is_visible()


def test_settings_modal_closes(page: Page):
    """設定モーダルの閉じるボタンで閉じられる。"""
    page.locator("#btn-open-settings").click()
    page.locator("#transcription-modal").wait_for(state="visible")
    page.locator("#btn-close-modal").click()
    page.locator("#transcription-modal").wait_for(state="hidden")
    assert not page.locator("#transcription-modal").is_visible()


def test_pyannote_shows_hf_token_field(page: Page):
    """diarization-method を pyannote に切り替えると HF トークン欄が表示される。"""
    page.locator("#btn-open-settings").click()
    page.locator("#transcription-modal").wait_for(state="visible")

    page.locator("#diarization-method").select_option("pyannote")
    assert page.locator("#hf-token-item").is_visible()

    # speechbrain に戻すと非表示
    page.locator("#diarization-method").select_option("speechbrain")
    assert not page.locator("#hf-token-item").is_visible()

    page.locator("#btn-close-modal").click()


@pytest.mark.slow
def test_transcription_shows_result(page: Page):
    """文字起こし完了後にセグメントテーブルが表示される。

    注意: このテストは実際の Whisper を実行するため低速。
    CI では ``-m 'not slow'`` で除外すること。
    """
    mock_api_method(
        page,
        "select_file",
        {"success": True, "filename": "test.m4a", "path": "test.m4a"},
    )
    page.locator("#btn-select-file").click()
    page.wait_for_function(
        "document.getElementById('file-name').textContent.includes('test.m4a')"
    )
    page.locator("#btn-start").click()
    # 文字起こし完了を最大 120 秒待つ
    page.wait_for_function(
        "document.getElementById('segment-table-body').querySelectorAll('tr').length > 0",
        timeout=120_000,
    )
    assert page.locator("#segment-table-body tr").count() > 0


def test_progress_area_visible_during_transcription(page: Page):
    """文字起こし中は進捗エリアが表示され、完了後に非表示になる。"""
    # transcribe をゆっくり返すスタブ（Promise delay）
    page.evaluate("""
        selectedFile = 'dummy.m4a';
        document.getElementById('btn-start').disabled = false;
        pywebview.api.transcribe = () =>
            new Promise(resolve =>
                setTimeout(() =>
                    resolve({
                        success: true,
                        num_segments: 0,
                        num_speakers: 1,
                        language: 'ja',
                        speakers: [],
                        text: ''
                    }), 800)
            );
        pywebview.api.get_preview_segments = () =>
            Promise.resolve({ success: true, segments: [] });
    """)
    page.locator("#btn-start").click()
    # 進捗エリアが表示されることを確認
    page.locator("#progress-area").wait_for(state="visible", timeout=3000)
    # 完了後に非表示
    page.locator("#progress-area").wait_for(state="hidden", timeout=5000)


def test_text_input_modal_opens_and_closes(page: Page):
    """テキスト入力モーダルを開閉できる。"""
    page.locator("#btn-open-text-input").click()
    page.locator("#text-input-modal").wait_for(state="visible")
    assert page.locator("#text-ms-per-char").input_value() == "100"

    page.locator("#btn-cancel-text-input").click()
    page.locator("#text-input-modal").wait_for(state="hidden")


def test_create_transcription_result_from_multiline_text(page: Page):
    """貼り付けた各行が連続するセグメントとして一覧に表示される。"""
    page.evaluate("""
        window.__textImportArgs = null;
        const importedSegments = [
            {start: 0.0, end: 0.5, text: 'こんにちは', speaker: 'Speaker 1'},
            {start: 0.5, end: 0.7, text: '世界', speaker: 'Speaker 2'},
        ];
        pywebview.api.import_text = (text, msPerChar, settings) => {
            window.__textImportArgs = {text, msPerChar, settings};
            return Promise.resolve({
                success: true,
                text: '',
                num_segments: 2,
                num_speakers: 2,
                language: 'text',
                input_type: 'text',
                speakers: [{
                    name: 'Speaker 1',
                    sample_text: 'こんにちは',
                    segment_count: 1,
                    first_start: 0.0,
                    first_end: 0.5,
                }, {
                    name: 'Speaker 2',
                    sample_text: '世界',
                    segment_count: 1,
                    first_start: 0.5,
                    first_end: 0.7,
                }],
            });
        };
        pywebview.api.get_preview_segments = () => Promise.resolve({
            success: true,
            segments: importedSegments,
        });
    """)

    page.locator("#btn-open-text-input").click()
    page.locator("#text-input-content").fill("[Speaker1] こんにちは\n[Speaker 2]: 世界")
    page.locator("#btn-create-from-text").click()

    page.wait_for_function(
        "document.querySelectorAll('#segment-table-body tr').length === 2"
    )
    args = page.evaluate("window.__textImportArgs")
    assert args["text"] == "[Speaker1] こんにちは\n[Speaker 2]: 世界"
    assert args["msPerChar"] == 100

    rows = page.locator("#segment-table-body tr")
    assert rows.nth(0).locator(".col-text").inner_text() == "こんにちは"
    assert "Speaker 1" in rows.nth(0).locator(".col-speaker").inner_text()
    assert "00:00.0 → 00:00.5" in rows.nth(0).locator(".col-time").inner_text()
    assert rows.nth(1).locator(".col-text").inner_text() == "世界"
    assert "Speaker 2" in rows.nth(1).locator(".col-speaker").inner_text()
    assert "00:00.5 → 00:00.7" in rows.nth(1).locator(".col-time").inner_text()
    assert page.locator("#file-name").inner_text() == "テキスト入力（無音）"
    assert "テキスト入力" in page.locator("#result-stats").inner_text()
    assert page.locator("#btn-save").is_enabled()
    assert page.locator("#btn-save-project").is_enabled()
    assert not page.locator("#text-input-modal").is_visible()


def test_millisecond_timeline_formatting(page: Page):
    """10ms以下の時刻も一覧・編集欄用に失わず整形する。"""
    assert page.evaluate("formatTimeDetailed(0.01)") == "00:00.010"
    assert page.evaluate("formatTimeDetailed(0.1)") == "00:00.1"
    assert page.evaluate("formatTimeInputValue(0.001)") == "0.001"
    assert page.evaluate("formatTimeInputValue(3.5)") == "3.50"
