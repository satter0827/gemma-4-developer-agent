# Easy課題の実験

以下はローカルCPUでの準備と検証です。Notebookの起動や正式提出は含みません。

```sh
make setup
make cli ARGS='validate-experiment --config configs/evaluation/easy-sequential-v2.yaml'
make cli ARGS='prepare-experiment-tasks --config configs/evaluation/easy-sequential-v2.yaml'
make check
make cli ARGS='freeze-experiment --config configs/evaluation/easy-sequential-v2.yaml'
```

最後のコマンドが返す `preparation.json` を次の入力にします。固定後にソースや関連入力を変更した場合は、再度固定してください。過去の記録は上書きしません。

```sh
make cli ARGS='prepare-kaggle --name easy-sequential-cpu --experiment-report artifacts/experiments/実行ID/preparation.json --verify-only'
make cli ARGS='verify-experiment --bundle artifacts/kaggle/easy-sequential-cpu --sdk-config configs/environments/sdk-cpu.yaml'
```

公式CPU環境がなければ、先に `prepare-runtime-checks --config configs/environments/sdk-cpu.yaml` を実行します。CPU検証はGPUライブラリの読み込みと通信を拒否します。課題の未修正・参照修正は課題専用環境の保存済み対照検証を再照合し、公式SDK環境では同じ候補をコンパイルして合成応答を確認します。両環境の依存は混ぜません。Kaggleの実行経路では公式ハーネスの対照検証をモデル起動前に改めて行います。確認用12問の環境準備は初回実験に含みません。

後のGPU用コード生成には、返された `session.json` と、観測したquotaのJSONが必要です。quotaは `consumed`（対象予算内の消費時間）、`remaining`（公式画面の残時間）、`completed_runs`、`active` を指定します。不明な値を0にしません。Skill比較v3は開発予算7時間・予備1時間・追加1回90分です。既存消費2.830299時間と2実行を引き継ぎ、追加quotaの上限3時間を確認します。コード生成自体はNotebookをアップロード・起動しません。生成から起動までに枠や実行状態が変わった場合は再確認してください。

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

## 未使用課題の確認とSkill比較

```sh
make cli ARGS='plan-experiment --config configs/tasks/rotation.yaml'
make cli ARGS='validate-experiment --config configs/evaluation/easy-sequential-v2.yaml'
make cli ARGS='validate-experiment --config configs/evaluation/easy-skills-v3.yaml'
make cli ARGS='prepare-experiment-tasks --config configs/evaluation/easy-skills-v3.yaml'
```

環境補正後のSkill比較にはv3を使います。Skill v1と逐次構成の定義は過去の条件として保持し、逐次構成の実モデル評価は行いません。Skill比較は別の仮説であり、逐次構成や最適化と同時に変更しません。

巡回コマンドは次の未使用課題群を提案し、宣言済みのIDや順序を変更しません。`matches_declared_selection`がfalseなら新しい提案として扱い、既に固定した比較対象へ適用しません。Skill v3はv1・v2と同じ6問を引き継ぐため、旧IDの予約がある状態でも固定6問を維持します。CPU対照が成立しなければ準備失敗として残し、成績を付けたり補欠と交換したりしません。候補固定とSDK検証は前述と同じ経路です。Skill資産の変更でも新しい候補とCPU記録が必要になります。

## 公開記録の作成

レビュー用YAMLに次の項目を記入します。原因の確度は`confirmed`、`hypothesis`、`unknown`から選びます。観測だけで原因を確認済みにしません。

```yaml
record_id: easy-skills-result-v1
interpretation: 同じ課題と固定条件で比較した結果と限界を一文で記載する。
decision: inconclusive
parent_candidates:
  workflow-v1: workflow-v1
  workflow-skills-v1: workflow-v1
assessments:
  - category: verification
    certainty: hypothesis
    basis: 公式成否と実際の検証行動に照らして、原因と残る不確実性を記載する。
prompts:
  - prompt_id: workflow-v1
    path: agent/prompts/workflow.md
```

```sh
make cli ARGS='summarize-experiment --session artifacts/results/実行ID/session.json --review artifacts/review.yaml --record-output docs/records/experiments/easy-skills-result-v1.json'
```

数値は照合済みの証拠から作られます。同じ公開ファイルは上書きできません。採用しない結果も、未完了や未確認を含めて判断を残します。分析本文はコマンドから更新しません。

## GEPAのCPU準備と検証

```sh
make cli ARGS='prepare-optimization-environment --config configs/environments/gepa.yaml'
make cli ARGS='validate-optimization --config configs/evaluation/easy-gepa-v2.yaml'
make cli ARGS='prepare-experiment-tasks --config configs/evaluation/easy-sequential-v2.yaml'
make check
make cli ARGS='prepare-optimization --config configs/evaluation/easy-gepa-v2.yaml'
make cli ARGS='prepare-kaggle --name easy-gepa-cpu --optimization-report artifacts/optimizations/実行ID/preparation.json --verify-only'
make cli ARGS='verify-optimization --bundle artifacts/kaggle/easy-gepa-cpu --sdk-config configs/environments/sdk-cpu.yaml'
```

GEPA用Wheelは準備コマンドが公式PyPIから取得し、`artifacts/reference/optimization-wheels/`へ保存して設定のハッシュと照合します。依存の準備は通信を伴いますが、CPU検証中は通信とGPUライブラリの読み込みを拒否します。検証は実ライブラリの探索と固定SDKの候補コンパイルを、合成応答で確認します。解決率の確認には使えません。

最適化は比較用の生成5分・30ツール呼出し・50ターン、採点5分を固定します。起動20分、8評価で88分、指示生成2分、CPU準備10分、後始末5分の上限は合計125分です。GPU予算の初期値は未設定です。実行する場合は、最適化設定へ必要時間以上の`gpu_budget`を明示し、既存消費約2.830時間を含めたquotaを確認してください。設定変更後は新しい準備・CPU検証が必要です。候補を採用する場合は、指示を自作候補として新しいIDへ固定し、通常の事前検証と独立した比較を行います。

最適化の`gpu_budget`を設定する場合は、`job_timeout_seconds`（全体の秒数）、`quota_hours`（対象予算の時間）、`reserve_hours`（残す時間）、`quota_per_wall_hour`（壁時計1時間当たりの消費係数）、`max_runs`（実行回数）をすべて記入します。初期の4時間枠へ上限125分の実行を足せるとは限りません。時間の見積もり、観測した消費・残枠、予備を照合し、許可されない場合は実行しません。

## 起動コードのサイズ制限

KaggleのSaveKernelは1MB以上のソースを拒否します。生成時はUTF-8で1,000,000バイト未満を上限とし、超過した場合は生成先を作成せず停止します。小さいバンドルはLZMAで埋め込みます。

大きいバンドルには`prepare-kaggle`の`--bundle-dataset OWNER/PRIVATE_DATASET`を指定します。この指定だけではアップロードも実行も行いません。生成先の`source-bundle.zip`を、別のアップロード用ディレクトリへ`source-bundle.bin`として同一バイト列でコピーします。`dataset-metadata.json`のIDは指定したDatasetに揃え、`kaggle datasets create -p UPLOAD_DIRECTORY`でPrivateとして登録します。`--public`は指定しません。ZIPの自動展開を避けるため、転送名は`.bin`です。

Datasetはソースハッシュごとに新しいIDを使い、作成後は更新しません。Privateであることと、取得した`source-bundle.bin`のSHA-256が生成記録の`source_sha256`に一致することを確認してからNotebookを起動します。Notebookは入力を読み込んだ後にもハッシュを照合します。CPUとGPUで同じバンドルを使い、候補ZIPを再生成しません。
