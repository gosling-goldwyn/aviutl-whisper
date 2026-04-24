# UC-02: 音声ファイル選択

## 概要

ユーザーが音声ファイルを選択し、UI に反映されることを確認する。

---

## ユースケース一覧

### UC-02-01: ファイル選択ダイアログの呼び出し

**前提条件**: アプリが起動している

**操作**: 「ファイルを選択」ボタンをクリックする

**期待結果**:
- `pywebview.api.select_file()` が呼び出される
- ネイティブのファイル選択ダイアログが開く（E2E テストではモック）

### UC-02-02: ファイル選択後の UI 更新

**前提条件**: アプリが起動している

**操作**:
1. `pywebview.api.select_file` をモックして任意の音声ファイルパスを返す
2. 「ファイルを選択」ボタンをクリックする

**期待結果**:
- ファイル名ラベル (`#file-name`) にファイル名が表示される
- ファイル情報エリア (`#file-info`) にサイズ等の情報が表示される
- 「▶ 開始」ボタンが **有効** になる
- 「💾 プロジェクトを保存」ボタンは依然 **無効** (文字起こし前)

### UC-02-03: 対応音声形式の確認

**期待結果**:
- `.m4a`, `.mp3`, `.wav`, `.flac`, `.ogg`, `.aac`, `.wma` の各拡張子が受け付けられる
- ファイルダイアログのフィルタに「音声ファイル (*.m4a;*.mp3;*.wav;...)」が含まれる

### UC-02-04: ファイル選択キャンセル

**前提条件**: アプリが起動している

**操作**:
1. `pywebview.api.select_file` をモックして `null` を返す (キャンセル相当)
2. 「ファイルを選択」ボタンをクリックする

**期待結果**:
- ファイル名ラベルが変化しない（「未選択」のまま）
- 「▶ 開始」ボタンが依然 **無効**

### UC-02-05: ヘッダーの「📂 プロジェクトを開く」ボタン

**前提条件**: アプリが起動している

**操作**: 左ペインの「📂 プロジェクトを開く」ボタンをクリックする

**期待結果**:
- `pywebview.api.load_project()` が呼び出される（または未保存確認ダイアログ経由）
- ファイル選択ダイアログが開く（E2E テストではモック）

---

## テスト実装メモ

```python
def test_file_selection(page):
    # select_file をモック
    page.evaluate("""
        window.__origSelectFile = pywebview.api.select_file;
        pywebview.api.select_file = async () => ({
            path: 'C:\\\\test\\\\sample.m4a',
            name: 'sample.m4a',
            extension: '.m4a',
            size: 1024000
        });
    """)
    page.click("#btn-select-file")
    page.wait_for_function("document.getElementById('file-name').textContent !== '未選択'")
    assert page.locator("#file-name").inner_text() == "sample.m4a"
    assert not page.locator("#btn-start").is_disabled()
```
