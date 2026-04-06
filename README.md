# aviutl-whisper

m4a等の音声ファイルから、Whisperによる文字起こし＋話者分離を行うGUIツールです。

## 機能

- **音声文字起こし**: faster-whisper による高速・高精度な文字起こし
- **話者分離**: speechbrain による自動話者識別
- **複数出力形式**: SRT字幕 / CSV / TSV / プレーンテキスト
- **GPU/CPU自動検出**: CUDA GPUがあれば自動で高速処理

## 必要環境

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- ffmpeg（音声変換に必要）

## セットアップ

```bash
# 依存関係インストール
uv sync

# 実行
uv run python main.py

# デバッグモードで実行
uv run python main.py --debug
```

## 対応音声形式

m4a, mp3, wav, flac, ogg, aac, wma

## 出力形式

| 形式 | 拡張子 | 説明 |
|------|--------|------|
| テキスト | .txt | `[00:00 - 00:05] Speaker 1: テキスト` |
| SRT | .srt | 字幕ファイル標準形式 |
| CSV | .csv | カンマ区切り |
| TSV | .tsv | タブ区切り |