# UC-09: エクスポート

## 概要

文字起こし結果を各形式（EXO / SRT / CSV / TSV / テキスト）でファイルに保存する操作を確認する。

---

## ユースケース一覧

### UC-09-01: 文字起こし前は保存ボタンが無効

**前提条件**: アプリが起動している（文字起こし未実行）

**期待結果**:
- 「💾 exoファイルに保存」ボタンが **無効** (disabled)

### UC-09-02: 文字起こし後に保存ボタンが有効化

**前提条件**: モックデータが注入済み

**期待結果**:
- 「💾 exoファイルに保存」ボタンが **有効**

### UC-09-03: EXO ファイルの保存

**前提条件**:
- モックデータが注入済み
- `pywebview.api.save_result` をモックしてパスを返す設定済み

**操作**: 「💾 exoファイルに保存」ボタンをクリックする

**期待結果**:
- `pywebview.api.save_result("exo", exo_settings, speaker_mapping)` が呼び出される
- ファイル選択ダイアログ（保存先）が開く

### UC-09-04: 保存成功後のフィードバック

**前提条件**: `pywebview.api.save_result` がモックで成功レスポンスを返す設定済み

**操作**: 「💾 exoファイルに保存」ボタンをクリックする

**期待結果**:
- 成功メッセージまたは通知が表示される（またはエラーが表示されない）

### UC-09-05: 保存ダイアログキャンセル

**前提条件**:
- モックデータが注入済み
- `pywebview.api.save_result` がモックで `null` を返す（キャンセル相当）

**操作**: 「💾 exoファイルに保存」ボタンをクリックする

**期待結果**:
- エラーが表示されない（キャンセルは正常動作）
- UI の状態が変化しない

### UC-09-06: EXO 保存時に最新の設定が使用される

**前提条件**: モックデータが注入済み

**操作**:
1. フォントサイズを `48` に変更する
2. 「💾 exoファイルに保存」ボタンをクリックする

**期待結果**:
- `save_result` に渡される `exo_settings` の `font_size` が `48` である

### UC-09-07: 話者マッピング適用後のエクスポート

**前提条件**:
- モックデータが注入済み
- 話者マッピングで話者が入れ替えられている

**操作**: 「💾 exoファイルに保存」ボタンをクリックする

**期待結果**:
- `save_result` に渡される `speaker_mapping` がマッピング内容を反映している

---

## テスト実装メモ

```python
def test_save_button_enabled_after_transcription(page, inject_mock_segments):
    assert not page.locator("#btn-save").is_disabled()

def test_save_result_called_with_exo_settings(page, inject_mock_segments):
    called_args = []
    page.evaluate("""
        window.__origSaveResult = pywebview.api.save_result;
        pywebview.api.save_result = async (...args) => {
            window.__saveResultArgs = args;
            return { success: true, path: 'C:\\\\output.exo' };
        };
    """)
    page.fill("#exo-font-size", "48")
    page.click("#btn-save")
    args = page.evaluate("window.__saveResultArgs")
    assert args[0] == "exo"
    assert args[1]["font_size"] == 48
```
