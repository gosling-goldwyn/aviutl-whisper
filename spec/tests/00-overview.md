# E2E テスト概要

## 目的

aviutl-whisper の GUI を含めたエンドツーエンドのテストを自動化し、ユーザーが体験する実際のワークフローが正しく動作することを保証する。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│  pytest + playwright (テストランナー)                  │
│                                                     │
│  conftest.py                                        │
│    ├─ app_process fixture                           │
│    │    subprocess で main.py を起動                 │
│    │    WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=       │
│    │    --remote-debugging-port=9222                │
│    └─ page fixture                                  │
│         connect_over_cdp("http://localhost:9222")   │
└───────────────────┬─────────────────────────────────┘
                    │ Chrome DevTools Protocol (CDP)
┌───────────────────▼─────────────────────────────────┐
│  pywebview (WebView2)                               │
│    ├─ Python backend: Api class (api.py)            │
│    └─ Frontend: index.html / app.js / style.css     │
└─────────────────────────────────────────────────────┘
```

## テスト戦略

| カテゴリ | 対象 | 手法 |
|---|---|---|
| UI インタラクション | ボタン・フォーム・ナビゲーション | CDP 経由で実アプリを操作 |
| 文字起こし後の状態 | セグメント編集・プレビュー | `page.evaluate()` でモックデータを JS に注入 |
| ファイルダイアログ | `select_file` / `save_result` | JS グローバルに差し込んだモック関数で代替 |
| 重い処理 (Whisper 等) | `transcribe()` | `@pytest.mark.slow` でマークし CI ではスキップ |

## ディレクトリ構成

```
spec/tests/                          ← 本フォルダ（ユースケース定義）
tests/
  test_backend.py              ← 既存の単体テスト
  e2e/
    conftest.py                ← アプリ起動 + CDP 接続フィクスチャ
    helpers.py                 ← 共通ヘルパー関数
    fixtures/
      segments.json            ← モックセグメントデータ
    test_app_startup.py
    test_file_selection.py
    test_transcription.py
    test_segment_editing.py
    test_exo_settings.py
    test_preview.py
    test_speaker_settings.py
    test_project_management.py
    test_export.py
    test_undo_redo.py
    test_keyboard_shortcuts.py
```

## セットアップ手順

```bash
# 依存関係追加
uv add --dev playwright pytest-playwright

# Chromium バイナリ取得
.venv\Scripts\playwright install chromium

# E2E テスト実行 (slow テストを除く)
.venv\Scripts\python -m pytest tests/e2e/ -m "not slow" -v

# 全テスト実行
.venv\Scripts\python -m pytest tests/e2e/ -v
```

## 前提条件

- Windows 環境 (WebView2 依存)
- WebView2 Runtime がインストール済みであること
- `uv sync` で依存関係がインストール済みであること
- `@pytest.mark.slow` テストはモデルダウンロード済み環境でのみ実行

## モックデータ仕様

文字起こし後の UI テストでは以下のモックセグメントデータを使用する:

```json
{
  "segments": [
    { "start": 0.0,  "end": 3.5,  "speaker": "Speaker 1", "text": "こんにちは、テストです。" },
    { "start": 3.8,  "end": 7.2,  "speaker": "Speaker 2", "text": "はじめまして。よろしくお願いします。" },
    { "start": 7.5,  "end": 12.0, "speaker": "Speaker 1", "text": "今日はよい天気ですね。" }
  ],
  "num_speakers": 2,
  "language": "ja"
}
```
