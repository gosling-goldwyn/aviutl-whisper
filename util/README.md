# util/headless_mp4_export.py — ヘッドレスMP4書き出し

aviutl-whisper のプロジェクト（`.awproj`）から、GUIを使わずにMP4を書き出す
ユーティリティです。

本アプリのMP4出力は Windows の GUI（WebView2 + WebCodecs）の中で動きます。
Linux サーバーや CI のように GUI を起動できない環境では同じ書き出しが
できないため、リポジトリの仕様をもとに描画・音声ミックスを再実装しました。

準拠した仕様・実装:

- `spec/mp4-export.md`（Canvas描画、エンコードパラメータ）
- `spec/exo-export.md`（BGM・ジングル・クロージングのタイムライン規則）
- `src/aviutl_whisper/web/canvas-renderer.js`（タイムライン計算、立ち絵の
  カラー/グレースケール切替、字幕の折り返しと配置）

## 使い方

```bash
pip install -r requirements.txt   # Pillow / numpy / fonttools
python3 headless_mp4_export.py プロジェクト.awproj \
    --media-dir /path/to/media \
    --out output.mp4
```

- `--media-dir` は複数回指定できます。プロジェクトや exo_settings が参照する
  ファイル（Windowsパス）は、ファイル名でこのディレクトリ内から探します。
- 本編音声はプロジェクトの `source_file` をファイル名で解決します。
  別のファイルを使う場合は `--audio` で指定してください。
- 字幕フォントは exo_settings のフォント名をシステムフォントから探します。
  見つからない字だけ Noto Sans CJK JP にフォールバックします。
  `--font FONT.ttf` で直接指定もできます。
- その他のオプション: `--seed`（BGMシャッルの種、既定は固定値で再現可能）、
  `--bitrate`（既定 `8M`、仕様の1080p30基準）、`--keep-work`、`--no-encode`。

## 動作環境

- Python 3.10+
- ffmpeg / ffprobe（PATH 上にあること）
- Pillow, numpy, fonttools

## 仕様どおり再現しているもの

- タイムライン: 開始オフセット（`audio_start_seconds`）、本編尺=
  max(音声尺, 最終字幕終了) のフレーム量子化、アウトロ、クロージング
- 映像: 背景 cover 配置、立ち絵（発話中カラー／非発話中グレースケール、
  イントロとアウトロ区間は全員カラー）、話者色字幕（20字折り返し等は
  exo_settings に従う）、クロージング画像は原寸・中央配置
- 音声: 本編音声100%、BGM は開始曲 → リストをシャッフル再生（連続同曲回避）、
  終了曲は本編終了30秒前から（直前を1.5秒フェード）、`bgm_volume` 適用、
  アウトロでフェードアウト、ジングルは100%・繰り返しなし
- 出力: H.264（既定8Mbps）/ AAC 48kHz ステレオ 192kbps、faststart

最適化として、映像は「見た目の状態」が変わる境界（セグメント開始・終了）
でのみ描画し、同一状態のフレーム列は ffmpeg の concat 入力で尺を持たせて
います。全フレームを描き直すより大幅に高速です。

## 字幕の分割（長文セグメント）

1つのセグメントが長いと字幕が何行にもなって読みにくいため、表示時に
自動で分割します。

- 折り返し（`max_chars_per_line`）で **3行を超える**セグメントは、
  文節の切れ目（。！？、）を優先して複数チャンクに分割し、時間は
  文字数比例で割り当てます
- 1チャンクは最大でも **4行** まで（`--soft-lines` / `--hard-lines`、
  または exo_settings の `subtitle_soft_lines` / `subtitle_hard_lines` で変更可）
- 分割は **単語の途中では行いません**。日本語の単語境界は janome で判定します
  （janome が無い環境では簡易的な文字種境界にフォールバックします）

## 制約・未対応

- `audio_mode` が `source` のプロジェクトのみ対応（結合TTS音声は未対応）
- `display_speed` が正の字幕（タイプライター表示）は未対応
- Canvas とピクセル単位で同一ではありません（フォントのラスタライズや
  縁取りの描き方が Pillow と Canvas で異なるため）
- BGMシャッフルの並びはアプリ本体と一致しません（種は固定でき、再現性はあります）
