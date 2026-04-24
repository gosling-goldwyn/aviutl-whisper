"""UC-04: セグメント編集テスト"""

import pytest
from playwright.sync_api import Page

from tests.e2e.helpers import (
    get_preview_nav_text,
    get_segment_table_rows,
    mock_api_method,
)

pytestmark = pytest.mark.e2e


class TestSegmentTableDisplay:
    def test_segment_table_shows_all_rows(self, mock_segments: Page):
        """モックセグメント注入後、セグメントテーブルに 3 行表示される。"""
        rows = get_segment_table_rows(mock_segments)
        assert len(rows) == 3

    def test_first_row_is_active(self, mock_segments: Page):
        """初期状態では 1 行目がアクティブ。"""
        rows = get_segment_table_rows(mock_segments)
        assert rows[0]["active"] is True

    def test_speaker_column_content(self, mock_segments: Page):
        """話者列に Speaker 1 / Speaker 2 が含まれる。"""
        rows = get_segment_table_rows(mock_segments)
        speakers = [r["speaker"] for r in rows]
        assert any("Speaker 1" in s for s in speakers)
        assert any("Speaker 2" in s for s in speakers)

    def test_text_column_content(self, mock_segments: Page):
        """テキスト列に期待するテキストが含まれる。"""
        rows = get_segment_table_rows(mock_segments)
        texts = [r["text"] for r in rows]
        assert "こんにちは、テストです。" in texts

    def test_time_column_format(self, mock_segments: Page):
        """時刻列が 'mm:ss.s → mm:ss.s' 形式。"""
        rows = get_segment_table_rows(mock_segments)
        import re
        pattern = r"\d{2}:\d+\.\d → \d{2}:\d+\.\d"
        assert re.match(pattern, rows[0]["time"])


class TestSegmentTableNavigation:
    def test_click_row_changes_active(self, mock_segments: Page):
        """セグメントテーブルの行をクリックするとアクティブ行が変わる。"""
        mock_segments.locator("#segment-table-body tr").nth(1).click()
        rows = get_segment_table_rows(mock_segments)
        assert rows[1]["active"] is True
        assert rows[0]["active"] is False

    def test_click_row_updates_preview_nav(self, mock_segments: Page):
        """2 行目をクリックするとプレビューナビゲーションが '2 / 3' に変わる。"""
        mock_segments.locator("#segment-table-body tr").nth(1).click()
        assert get_preview_nav_text(mock_segments) == "2 / 3"


class TestSegmentEditorPanel:
    def test_editor_shows_current_segment(self, mock_segments: Page):
        """セグメントエディターに現在のセグメントのテキストが表示される。"""
        assert mock_segments.locator("#seg-editor").is_visible()
        assert mock_segments.locator("#seg-edit-text").input_value() == "こんにちは、テストです。"

    def test_editor_start_end_values(self, mock_segments: Page):
        """開始・終了時刻フィールドに正しい値が表示される。"""
        assert mock_segments.locator("#seg-edit-start").input_value() == "0.00"
        assert mock_segments.locator("#seg-edit-end").input_value() == "3.50"

    def test_merge_prev_disabled_on_first_segment(self, mock_segments: Page):
        """先頭セグメントでは '前と結合' ボタンが無効。"""
        assert mock_segments.locator("#btn-seg-merge-prev").is_disabled()

    def test_merge_next_disabled_when_different_speaker(self, mock_segments: Page):
        """次のセグメントが異なる話者の場合は '次と結合' ボタンが無効。"""
        # Speaker 1 (idx=0) → next is Speaker 2 (idx=1)
        assert mock_segments.locator("#btn-seg-merge-next").is_disabled()


class TestSegmentUpdate:
    def test_apply_segment_edit_updates_table(self, mock_segments: Page):
        """セグメント編集を適用するとテーブルが更新される。"""
        page = mock_segments
        mock_api_method(
            page,
            "update_segment",
            {
                "success": True,
                "segments": [
                    {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": "編集後のテキスト"},
                    {"start": 3.8, "end": 7.2, "speaker": "Speaker 2", "text": "はじめまして。よろしくお願いします。"},
                    {"start": 7.5, "end": 12.0, "speaker": "Speaker 1", "text": "今日はよい天気ですね。"},
                ],
            },
        )
        page.locator("#seg-edit-text").fill("編集後のテキスト")
        page.locator("#btn-seg-apply").click()
        page.wait_for_timeout(300)

        rows = get_segment_table_rows(page)
        assert rows[0]["text"] == "編集後のテキスト"

    def test_merge_segments_reduces_row_count(self, mock_segments: Page):
        """同じ話者の隣接セグメントを結合すると行数が 1 減る。"""
        page = mock_segments
        # 2 行目 (index=1, Speaker 2) から次の行 (Speaker 1) を結合する
        page.locator("#segment-table-body tr").nth(1).click()
        mock_api_method(
            page,
            "merge_segments",
            {
                "success": True,
                "merged_index": 1,
                "segments": [
                    {"start": 0.0, "end": 3.5, "speaker": "Speaker 1", "text": "こんにちは、テストです。"},
                    {"start": 3.8, "end": 12.0, "speaker": "Speaker 2", "text": "はじめまして。今日はよい天気ですね。"},
                ],
            },
        )
        page.evaluate("mergeNextSegment()")
        page.wait_for_timeout(300)
        rows = get_segment_table_rows(page)
        assert len(rows) == 2
