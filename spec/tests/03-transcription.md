# UC-03: 文字起こし設定・実行・キャンセル

## 概要

文字起こしの設定変更、実行開始、進捗表示、完了後の結果反映、およびキャンセルを確認する。

---

## ユースケース一覧

### UC-03-01: 文字起こし設定モーダルを開く

**前提条件**: アプリが起動している

**操作**: 「🔧 設定」ボタンをクリックする

**期待結果**:
- 文字起こし設定モーダル (`#transcription-modal`) が表示される
- モーダル内に以下の設定項目が存在する:
  - Whisperモデル選択 (`#model-size`): tiny / base / small / medium / large-v3
  - 言語選択 (`#language`): 自動検出 / 日本語 / 英語 / 中国語 / 韓国語
  - 話者数選択 (`#num-speakers`): 自動推定 / 1〜5人
  - 話者分離方式 (`#diarization-method`): speechbrain / pyannote

### UC-03-02: pyannote 選択時の HF トークン欄表示

**前提条件**: 文字起こし設定モーダルが開いている

**操作**: 「話者分離方式」を「pyannote (高精度)」に変更する

**期待結果**:
- 「HuggingFace トークン」入力欄 (`#hf-token-item`) が表示される

**操作**: 「話者分離方式」を「speechbrain (内蔵)」に戻す

**期待結果**:
- 「HuggingFace トークン」入力欄が非表示になる

### UC-03-03: 設定モーダルを閉じる (OK / 背景クリック / ✕ボタン)

**前提条件**: 文字起こし設定モーダルが開いている

**操作A**: 「OK」ボタンをクリックする  
**操作B**: モーダル背景（オーバーレイ）をクリックする  
**操作C**: 「✕」ボタンをクリックする

**期待結果** (いずれも):
- 設定モーダルが閉じる

### UC-03-04: 文字起こし開始

**前提条件**: 音声ファイルが選択済み

**操作**: 「▶ 開始」ボタンをクリックする

**期待結果**:
- 「▶ 開始」ボタンが非表示になり「⏹ 中止」ボタンが表示される
- 進捗バーエリア (`#progress-area`) が表示される
- 進捗テキスト (`#progress-text`) が更新される

### UC-03-05: 文字起こし完了後の UI 更新

**前提条件**: `pywebview.api.transcribe()` がモックで即座に成功レスポンスを返すように設定済み

**操作**: 「▶ 開始」ボタンをクリックし、完了を待つ

**期待結果**:
- 「⏹ 中止」ボタンが非表示になり「▶ 開始」ボタンが再表示される
- セグメント一覧テーブルに結果が表示される
- 「💾 exoファイルに保存」ボタンが **有効** になる
- 「💾 プロジェクトを保存」ボタンが **有効** になる
- 話者マッピングセクションが表示される

### UC-03-06: 文字起こしキャンセル

**前提条件**: 文字起こしが実行中

**操作**: 「⏹ 中止」ボタンをクリックする

**期待結果**:
- `pywebview.api.cancel()` が呼び出される
- 「⏹ 中止」ボタンが非表示になり「▶ 開始」ボタンが再表示される
- 進捗バーエリアが非表示になる

### UC-03-07: 文字起こしエラー時の表示

**前提条件**: `pywebview.api.transcribe()` がエラーレスポンスを返すようにモック設定済み

**操作**: 「▶ 開始」ボタンをクリックし、エラー完了を待つ

**期待結果**:
- エラーメッセージが表示される（アラートまたは画面上の通知）
- 「▶ 開始」ボタンが再び有効になる

---

## テスト実装メモ

```python
def test_transcription_modal_open(page):
    page.click("#btn-open-settings")
    assert page.locator("#transcription-modal").is_visible()

def test_pyannote_token_visibility(page):
    page.click("#btn-open-settings")
    page.select_option("#diarization-method", "pyannote")
    assert page.locator("#hf-token-item").is_visible()
    page.select_option("#diarization-method", "speechbrain")
    assert not page.locator("#hf-token-item").is_visible()

@pytest.mark.slow
def test_transcription_start(page):
    # 実際のモデルが必要なため slow マーク
    ...
```
