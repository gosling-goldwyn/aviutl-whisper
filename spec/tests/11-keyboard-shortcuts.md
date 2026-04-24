# UC-11: キーボードショートカット

## 概要

アプリが提供するキーボードショートカットの動作を確認する。

---

## ショートカット一覧

| ショートカット | 動作 |
|---|---|
| `Ctrl+Z` | Undo (テキストエリア外) |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |
| `Ctrl+S` | プロジェクトを保存 (上書き / 初回は別名) |
| `Ctrl+Shift+S` | プロジェクトを別名で保存 |
| `Ctrl+O` | プロジェクトを開く |
| `←` (左矢印) | 前のセグメントへ (テキストエリア外) |
| `→` (右矢印) | 次のセグメントへ (テキストエリア外) |

---

## ユースケース一覧

### UC-11-01: Ctrl+Z で Undo

**前提条件**: 編集操作が1件ある (Undo 可能な状態)

**操作**: `Ctrl+Z` を押す（テキストエリア外）

**期待結果**:
- Undo が実行される

### UC-11-02: Ctrl+Y で Redo

**前提条件**: Undo 後の状態 (Redo 可能)

**操作**: `Ctrl+Y` を押す

**期待結果**:
- Redo が実行される

### UC-11-03: Ctrl+Shift+Z で Redo

**前提条件**: Undo 後の状態

**操作**: `Ctrl+Shift+Z` を押す

**期待結果**:
- Redo が実行される

### UC-11-04: Ctrl+S でプロジェクト保存

**前提条件**:
- モックデータが注入済み
- `pywebview.api.save_project` をモック済み

**操作**: `Ctrl+S` を押す

**期待結果**:
- `pywebview.api.save_project()` または `pywebview.api.save_project_as()` が呼び出される

### UC-11-05: Ctrl+Shift+S でプロジェクトを別名保存

**前提条件**:
- モックデータが注入済み
- `pywebview.api.save_project_as` をモック済み

**操作**: `Ctrl+Shift+S` を押す

**期待結果**:
- `pywebview.api.save_project_as()` が呼び出される

### UC-11-06: Ctrl+O でプロジェクトを開く

**前提条件**:
- `pywebview.api.load_project` をモック済み

**操作**: `Ctrl+O` を押す

**期待結果**:
- プロジェクト読み込みが実行される（dirty 状態なら確認ダイアログが表示される）

### UC-11-07: 左矢印キーで前のセグメントへ

**前提条件**:
- モックデータが注入済み
- 2番目以降のセグメントが選択されている
- テキストエリア・INPUT・SELECT にフォーカスがない

**操作**: `←` キーを押す

**期待結果**:
- 前のセグメントが選択される
- プレビューが更新される

### UC-11-08: 右矢印キーで次のセグメントへ

**前提条件**:
- モックデータが注入済み
- 最後以外のセグメントが選択されている
- テキストエリア等にフォーカスがない

**操作**: `→` キーを押す

**期待結果**:
- 次のセグメントが選択される
- プレビューが更新される

### UC-11-09: テキストエリアフォーカス中は矢印キーがナビゲーションを発火しない

**前提条件**: テキストエリア (`#seg-edit-text`) にフォーカスがある

**操作**: `←` または `→` キーを押す

**期待結果**:
- セグメントナビゲーションは発生しない（テキストカーソルが移動するだけ）

### UC-11-10: INPUT にフォーカス中は Ctrl+Z がネイティブ動作

**前提条件**: テキスト系 INPUT または TEXTAREA にフォーカスがある

**操作**: `Ctrl+Z` を押す

**期待結果**:
- アプリの Undo は発火しない

---

## テスト実装メモ

```python
def test_arrow_navigation(page, inject_mock_segments):
    # 最初のセグメントを選択
    page.click("#segment-table-body tr:first-child")
    assert page.locator("#preview-seg-info").inner_text() == "1 / 3"
    # body にフォーカスを当てる
    page.locator("body").click()
    # 右矢印で次へ
    page.keyboard.press("ArrowRight")
    assert page.locator("#preview-seg-info").inner_text() == "2 / 3"
    # 左矢印で戻る
    page.keyboard.press("ArrowLeft")
    assert page.locator("#preview-seg-info").inner_text() == "1 / 3"

def test_ctrl_s_calls_save(page, inject_mock_segments):
    save_called = []
    page.evaluate("""
        pywebview.api.save_project = async () => {
            window.__saveCalled = true;
            return { success: true, path: 'test.json' };
        };
    """)
    page.keyboard.press("Control+s")
    called = page.evaluate("window.__saveCalled")
    assert called is True
```
