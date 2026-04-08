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

## pyannote (オプション)

高精度な話者分離に pyannote.audio を利用できます（オプション依存）。設定手順:

1. インストール
   - 推奨（リポジトリの optional deps を使う）: uv pip install ".[pyannote]"
   - または個別に: uv pip install pyannote.audio

2. Hugging Face トークンを取得
   - https://huggingface.co にサインアップし、Settings → Access Tokens で新しいトークン（read 権限）を作成してコピーします。

3. モデル利用条件の承諾
   - pyannote の話者ダイアライゼーションモデルはゲート付きです。
   - https://huggingface.co/pyannote/speaker-diarization-3.1 にアクセスし、サインインした上でモデルページの利用条件（Access / Accept）を承諾してください。

4. トークンの登録
   - GUI: 設定で「話者分離方式」を "pyannote" に切り替え、"HuggingFace トークン" 欄に貼り付けて保存します。Windows ではトークンがDPAPIで暗号化されます。
   - テスト/CI: プロジェクトルートに `.env` を作成し `HF_TOKEN=hf_xxx` を置くとテストが参照できます（`.env` は .gitignore に含まれます）。

5. GPUでの実行（任意）
   - GPUで高速化したい場合は、事前にCUDA対応のPyTorchをインストールしてください（https://pytorch.org の公式手順を参照）。GPUが利用可能なら自動でGPUへ移動します。

トラブルシューティング

- `import` / 互換性エラーが出る場合は pyannote のバージョンや依存関係（torch等）を確認してください。本アプリは pyannote 3.x / 4.x の戻り値形式に対応しています。
- pyannote をインストールできない、またはモデルにアクセスできない場合は、上記のモデルページで利用条件の承諾（Access）を行っているかを確認してください。

