# 評価手順

## 固定課題の再確認

```sh
make cli ARGS='qualify --config configs/tasks/comparison.yaml --fixed-selection'
```

固定済み6問を同じ順序で確認し、`artifacts/qualification/current.json`が新しい記録を指します。既存の`qualified.json`と過去のケースは保持します。失敗しても補欠へ入れ替えません。初回選定専用の`qualify`は既存の固定記録があれば拒否します。

## 候補を固定する

```sh
make cli ARGS='preflight --config configs/evaluation/preflight.yaml --profile baseline'
make cli ARGS='preflight --config configs/evaluation/preflight.yaml --profile workflow'
```

各出力の`report.json`を次の操作へ渡します。smokeは1分・10ツール、比較プロファイルは5分・30ツールです。比較課題を再検証してから候補を固定します。

## Notebookを生成する

```sh
make cli ARGS='prepare-kaggle --name comparison-cpu --preflight-report PREFLIGHT_REPORT --verify-only'
make cli ARGS='prepare-kaggle --name comparison-gpu --preflight-report PREFLIGHT_REPORT --cpu-report CPU_MANIFEST'
```

生成先は`artifacts/kaggle/名前`です。生成だけではKaggleへ送信しません。CPUとGPUで候補ZIPとソースは同じで、GPU用には同一候補の完了済みCPU証拠が必要です。配置変更後のコードには新しいCPU証拠を要求し、過去の証拠を現在のものに置き換えません。

遠隔実行を明示的に行うときだけ、生成ディレクトリを`kaggle kernels push -p`で送ります。Private・インターネット無効を維持し、CPU結果を確認してからGPUへ進みます。Notebookは45分で終了処理まで含めて制限します。事前に`kaggle quota --format json`で残枠と回復日時を記録し、`check-gpu-budget`で確認します。自動再試行や並行実行は行いません。

## 結果の確認

```sh
make cli ARGS='results'
make cli ARGS='show RUN_ID --task TASK_ID'
make cli ARGS='analyze RUN_ID --other OTHER_RUN_ID'
make cli ARGS='compare LEFT_RUN_ID RIGHT_RUN_ID'
```

公式判定、個別テスト、生成中の行動を分けて表示します。結果と派生分析はGit管理外です。欠損・重複・集計不一致を確認し、実行順序が一致しない比較は有効としません。Notebook評価と正式提出は独立しています。
