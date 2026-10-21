# Easy課題の実験

以下はローカルCPUでの準備と検証です。Notebookの起動や正式提出は含みません。

```sh
make setup
make cli ARGS='validate-experiment --config configs/evaluation/easy-sequential-v1.yaml'
make cli ARGS='prepare-experiment-tasks --config configs/evaluation/easy-sequential-v1.yaml'
make check
make cli ARGS='freeze-experiment --config configs/evaluation/easy-sequential-v1.yaml'
```

最後のコマンドが返す `preparation.json` を次の入力にします。固定後にソースや関連入力を変更した場合は、再度固定してください。過去の記録は上書きしません。

```sh
make cli ARGS='prepare-kaggle --name easy-sequential-cpu --experiment-report artifacts/experiments/実行ID/preparation.json --verify-only'
make cli ARGS='verify-experiment --bundle artifacts/kaggle/easy-sequential-cpu --sdk-config configs/environments/sdk-cpu.yaml'
```

公式CPU環境がなければ、先に `prepare-runtime-checks --config configs/environments/sdk-cpu.yaml` を実行します。CPU検証はGPUライブラリの読み込みと通信を拒否します。課題の未修正・参照修正は課題専用環境の保存済み対照検証を再照合し、公式SDK環境では同じ候補をコンパイルして合成応答を確認します。両環境の依存は混ぜません。Kaggleの実行経路では公式ハーネスの対照検証をモデル起動前に改めて行います。確認用12問の環境準備は初回実験に含みません。

後のGPU用コード生成には、返された `session.json` と、観測したquotaのJSONが必要です。quotaは `consumed`（対象予算内の消費時間）、`remaining`（公式画面の残時間）、`completed_runs`、`active` を指定します。不明な値を0にしません。既存消費を含む4時間の枠では90分セッションを追加できないため、明示的な予算判断が必要です。コード生成自体はNotebookをアップロード・起動しません。生成から起動までに枠や実行状態が変わった場合は再確認してください。

```sh
make cli ARGS='prepare-kaggle --name easy-sequential-gpu --experiment-report artifacts/experiments/実行ID/preparation.json --cpu-report artifacts/experiment-verification/実行ID/session.json --quota artifacts/quota.json'
make cli ARGS='summarize-experiment --session artifacts/results/実行ID/session.json'
```

完了した組だけを比較し、未完了・エラー・未実行を確認します。実際のquotaと全体時間も記録し、上限を超えて再試行しません。比較から得た判断は分析文書へ条件と根拠を直接記述します。

## MLflowへの同期と復旧

```sh
make cli ARGS='prepare-tracking --config configs/environments/mlflow.yaml'
make cli ARGS='sync-tracking --config configs/environments/mlflow.yaml --record docs/records/experiments/easy-sequential-cpu-v1.json'
```

SQLiteは `artifacts/mlflow/tracking.db` に保存します。同じ記録の再同期では重複を作りません。同じIDで内容が違えば停止するので、新しいIDで記録を追加します。破損時はDBを退避して同じGit記録を再同期します。プロンプトは自作の未展開指示だけを登録します。競技本文やログを手動で追加しないでください。

GPU実行前には、最新のCPU証拠、候補ハッシュ、固定した課題と順序、残枠、既存消費、同時実行がないこと、90分の期限と停止条件を確認します。GPU結果が得られるまで、逐次構成の精度は未評価です。
