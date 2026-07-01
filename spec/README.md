# 仕様書

このフォルダは製品・データ仕様とE2Eテスト仕様を分けて管理します。

- [project-format.md](project-format.md): `.awproj` プロジェクト形式
- [exo-export.md](exo-export.md): GUIから生成するEXOの規則
- [tests/00-overview.md](tests/00-overview.md): E2Eテストの前提と実行方法
- [tests/coverage.md](tests/coverage.md): ユースケースと自動テストの対応状況

`tests/*.md` のユースケースは期待する振る舞いを定義します。「自動化済み」であることは
意味しません。自動テストによる保証範囲は `tests/coverage.md` を正とします。
