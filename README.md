# aviutl-whisper

音声ファイルを faster-whisper で文字起こしし、話者分離した結果を AviUtl
拡張編集の `.exo` ファイルとして保存する Windows 向けGUIツールです。

## 主な機能

- faster-whisper による文字起こし（tiny / base / small / medium / large-v3）
- 改行区切りテキストからのセグメント作成と無音タイムライン生成
- speechbrain による話者分離、またはオプションの pyannote.audio による話者分離
- セグメントのテキスト・話者・開始時刻・終了時刻の編集、追加、削除、結合
- 話者ごとの字幕色・縁色・立ち絵と、背景画像を反映したシーンプレビュー
- AviUtl拡張編集用EXO出力（字幕、背景、立ち絵、非発話中の立ち絵のモノクロ化）
- `.awproj` プロジェクトの保存・再開、Undo / Redo、キーボードショートカット
- CUDA GPUの自動検出と、利用できない場合のCPUフォールバック

GUIから保存できる出力形式は `.exo` のみです。SRT / CSV / TSV / TXT の生成関数も
バックエンドにありますが、現在のGUIからは選択できません。

## 必要環境

- Windows 10 / 11
- Python 3.13以上
- [uv](https://docs.astral.sh/uv/)
- Microsoft Edge WebView2 Runtime
- ffmpeg（m4a、mp3などをWAVへ変換するために必要）

CUDA対応GPUは任意です。GPUを使う場合は、利用環境に合うCUDAドライバーが別途必要です。

## セットアップと起動

```powershell
uv sync
uv run python main.py
```

デバッグモードでは次のように起動します。

```powershell
uv run python main.py --debug
```

初回の文字起こし時には、選択したWhisperモデルとspeechbrainモデルをインターネットから
ダウンロードします。Whisperモデルの概算サイズは tiny: 75 MB、base: 140 MB、
small: 460 MB、medium: 1.5 GB、large-v3: 3 GBです。

モデルは `%LOCALAPPDATA%\aviutl-whisper\whisper` と
`%LOCALAPPDATA%\aviutl-whisper\speechbrain` にキャッシュされます。

## 基本的な使い方

1. 「ファイルを選択」で音声を選択します。
2. 「設定」でWhisperモデル、言語、話者数、話者分離方式を指定します。
3. 「開始」で文字起こしと話者分離を実行します。
4. セグメント、話者割り当て、字幕、背景、立ち絵を編集します。
5. プレビューを確認し、「exoファイルに保存」で出力します。

対応音声形式は m4a、mp3、wav、flac、ogg、aac、wma です。

音声を使わずに字幕を作る場合は「テキストから作成」を開き、改行区切りのテキストを
貼り付けます。各行が1セグメントになり、既定では1文字100msとして隙間なく配置されます。
1文字あたりの時間は作成前に変更できます。前後の空白と空行は除外され、タイムラインと
同じ長さの無音音声が使用されます。
行頭に `[Speaker1]` または `[Speaker 2]` を付けると、そのprefixを除いた本文へ
`Speaker 1` / `Speaker 2` を割り当てます。prefixのない行は直前の話者を継承し、
最初に話者指定がない場合は `Speaker 1` になります。

## プロジェクトと設定

作業状態は `.awproj` ファイルに保存できます。プロジェクトにはセグメント、話者割り当て、
EXO設定、元音声のパス、プレビュー位置が含まれます。元音声を移動・削除した後も編集と
EXO出力はできますが、セグメント音声の再生はできません。

アプリ設定は `%LOCALAPPDATA%\aviutl-whisper\settings.json` に自動保存されます。
Hugging FaceトークンはWindowsではDPAPIで暗号化されます。レイアウトの折りたたみ状態と
ペインサイズはWebView2のlocalStorageに保存されます。

プロジェクト形式の詳細は [spec/project-format.md](spec/project-format.md)、EXO生成規則は
[spec/exo-export.md](spec/exo-export.md) を参照してください。

## pyannote.audioを使う場合（オプション）

1. オプション依存をインストールします。

   ```powershell
   uv sync --extra pyannote
   ```

2. [Hugging Face](https://huggingface.co/) でread権限のトークンを作成します。
3. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   と、そのページから案内されるゲート付き依存モデルの利用条件に同意します。
4. 文字起こし設定で話者分離方式を「pyannote」にし、トークンを入力します。

テストやCIでは、リポジトリ直下の `.env` に `HF_TOKEN=hf_xxx` を設定できます。
`.env` はコミットしないでください。

## テスト

```powershell
# バックエンド単体テスト
uv run pytest tests/test_backend.py -q

# Windows + WebView2が必要なE2Eテスト
uv run pytest tests/e2e -m "not slow" -v
```

E2Eテストの前提とユースケースは [spec/tests/00-overview.md](spec/tests/00-overview.md) を
参照してください。

## 制約とトラブルシューティング

- 初回モデル取得にはインターネット接続と十分なディスク容量が必要です。
- CUDAでモデルを読み込めない場合はCPUへフォールバックするため、処理が遅くなります。
- pyannoteで403系エラーになる場合は、モデル利用条件への同意とトークン権限を確認してください。
- EXOはAviUtl互換のためCP932で保存されます。CP932で表現できないパスや設定値は避けてください。
- 音声変換に失敗する場合は、`ffmpeg -version` が実行できることを確認してください。
