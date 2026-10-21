# ログの確認

CLIのイベントは`artifacts/logs/<execution_id>/`、診断・評価・最適化のイベントは各実行先の`logs/events/`にあります。ファイル名のプロセスIDで親子を区別し、`execution_id`で同じ処理をまとめます。すべてJSON Linesです。

```sh
jq -c 'select(.level == "ERROR" or .level == "WARNING")' artifacts/logs/実行ID/*.jsonl
jq -c 'select(.task_id == "fastapi_14297")' artifacts/easy-rotation/controls/実行ID/logs/events/*.jsonl
jq -c 'select(.event == "tool.observed") | {task_id,tool_name,stage,result,reason_code}' artifacts/results/実行ID/logs/events/*.jsonl
```

`pyproject.toml`の`tool.gemma_agent_dev.logging`でファイルとコンソールのレベル、ローテーション容量、世代数を変更できます。詳細な照合を調べる場合はファイルをDEBUGにし、変更後のNotebookは作り直します。

ツールの所要時間がnullの場合は未確認です。課題が未解決でも、基盤が正常ならINFOになります。採点結果は公式の個別結果、候補の照合はハッシュ、提出状況は受付状態の記録で確認します。ログだけから再提出を判断しません。

保存不能が表示された場合は、空き容量と保存先の権限を直し、確定済みの採点結果・受付状態を先に確認してください。ログを補う目的で評価や提出を再実行しません。認証情報や課題本文をログへ追記せず、既存のSDKログやトレースを私用の保存先で確認します。イベントログはGit、MLflow、Actions成果物、提出ZIPへ追加しません。

各実行先の`logging-status.json`で`status`が`ok`なら保存障害は検出されていません。`degraded`は`reason_codes`に初期化・書き込み・終了・健全性保存の障害を示します。旧記録などでファイルがない場合は未確認です。健全性情報が欠損しても公式判定を失敗へ変更しません。ジョブの追跡には`job_id`、同一コードの照合にはソースハッシュを使い分けます。
