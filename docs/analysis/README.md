# 問題の分析と管理

固定された開発用129問を、共通の基準で整理します。目的は、次の評価で何を確認するか、そのためにどれだけGPU枠が必要かを判断できるようにすることです。今回の整備では評価対象やエージェントを変更しません。

- [分析基準](criteria.md)：構造指標、課題文の確認、成否、費用の読み方
- [問題台帳](tasks.md)：全問題の区分、構造指標、過去の公式判定
- [評価手順](../evaluation.md)：候補を固定してCPU・GPUで検証する方法
- [提出手順](../submission.md)：検証済み候補の登録と期限付き提出

全体129問、実行設定15問、既存比較6問、診断2問は別の区分です。実行設定にない問題も台帳へ含めます。比較対象は選定時の記録から読み、結果を見て入れ替えません。

準備済みCPU環境と保存済みの比較結果がある状態で生成します。

```sh
make cli ARGS='task-report --config configs/task-analysis.yaml'
make cli ARGS='task-report --config configs/task-analysis.yaml --write-docs'
```

通常は`artifacts/task-analysis/`へJSONとMarkdownを出力します。`--write-docs`を指定すると、このフォルダーの問題台帳も更新します。手動の確認内容は`configs/task-analysis.yaml`へ記入し、生成文書を直接編集しません。

公式SDKの版、Wheelのハッシュ、インストール済みPythonソースを照合します。集計中の通信・外部プロセス・GPUライブラリのimportは拒否します。問題文、正解パッチ、生成パッチ、実行ログは公開台帳へ出力しません。
