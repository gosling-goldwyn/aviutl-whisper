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
