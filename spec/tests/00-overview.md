# E2E テスト概要

## 目的

aviutl-whisper のGUIをWebView2上で操作し、主要ワークフローの回帰を検出する。
各ユースケースは期待仕様であり、自動化状況は [coverage.md](coverage.md) を正とする。

## アーキテクチャ

```text
pytest + Playwright
  └─ main.pyを非表示ウィンドウで起動
      └─ WebView2のCDPポートへconnect_over_cdp()
          ├─ Python: Api (api.py)
          └─ Frontend: index.html / app.js / style.css
```

E2Eは外部Chromiumを起動せず、実アプリのWebView2へ接続する。文字起こし結果や
ファイルダイアログは、対象テストに応じてJS側へモックを注入する。

## テスト戦略

| カテゴリ | 対象 | 手法 |
|---|---|---|
| UIインタラクション | ボタン、フォーム、ナビゲーション、レイアウト | CDP経由で実アプリを操作 |
| 文字起こし後の状態 | セグメント、プレビュー、話者設定 | `page.evaluate()` でモックデータを注入 |
| ファイルダイアログ | 選択、保存、読み込み | `pywebview.api` の対象メソッドをJSでモック |
| Whisperなどの重い処理 | 実モデルを使う処理 | 通常E2Eから分離し、必要な環境でのみ実行 |

## ディレクトリ構成

```text
spec/tests/
  00-overview.md
  01-app-startup.md ... 12-layout.md
  coverage.md
tests/
  test_backend.py
  e2e/
    conftest.py
    helpers.py
    fixtures/segments.json
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
    test_layout.py
```

## セットアップと実行

```powershell
uv sync
uv run pytest tests/e2e -m "not slow" -v
```

PlaywrightはCDPクライアントとして使うため、`playwright install chromium` は不要。
依存関係を変更する `uv add` もテスト環境のセットアップでは実行しない。

## 前提条件

- Windows
- Microsoft Edge WebView2 Runtime
- `uv sync` 済み
- CDP用のTCPポート9222が利用可能
- セッション中にWebView2ウィンドウを初期化できるデスクトップ環境

## モックデータ仕様

文字起こし後のUIテストでは `tests/e2e/fixtures/segments.json` を使用する。
各セグメントは `start`、`end`、`speaker`、`text` を持ち、現在のfixtureは
3セグメント、2話者、言語 `ja` を表す。
