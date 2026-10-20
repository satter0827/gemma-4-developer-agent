# Gemma 4 Developer Agent

KaggleのGemma 4 Developer Agent Competitionに向けた、提出エージェントと評価用CLIです。公式基盤を使用し、評価条件と提出物を追跡します。

## 開発環境

Python 3.12、uv、Gitを使用します。Kaggleでの実行には、認証済みのKaggle CLIとコンペ参加が必要です。

```sh
make setup
make check
```

コマンドはリポジトリのルートで実行します。開発環境はiCloud外に置きます。Makefileを通して実行すると、仮想環境、キャッシュ、一時ファイルはすべて`artifacts/`へ保存されます。通常のwheelとしてインストールし、コード変更後は`make setup`で更新します。`make check`は更新とすべての品質チェックを実行します。

テストでは行カバレッジ90%以上、分岐カバレッジ80%以上をそれぞれ必須にします。行の実行だけでなく、設定の拒否や失敗時の停止も確認するための最低率です。設定は`pyproject.toml`の`tool.coverage_gate`にまとめています。

計測対象は`src/gemma_agent_dev/`、`kaggle/`、`scripts/`です。開発用テストと外部ライブラリは対象に含めません。個別モジュールやエラー処理の除外は追加していません。[coverage.pyの分岐計測](https://coverage.readthedocs.io/en/latest/branch.html)を使い、JSONの行・分岐件数から別々に判定します。端末の`Cover`列は両方を合わせた率なので、合否は最後の`line`・`branch`の表で確認してください。

開発環境の準備後は`make coverage`でも再計測できます。`artifacts/coverage/`にJSON、HTML、判定表を出力し、毎回前の計測結果を消してから実行します。CIも`make check`で同じ基準を使い、判定表をジョブの概要へ、JSONとHTMLを`coverage`成果物へ保存します。閾値未達、分岐データなし、計測件数0はチェック失敗です。

## ローカルでパッチを診断する

保存済みパッチの失敗原因は、先にローカルのPython環境で確認します。

```sh
make cli ARGS='prepare-diagnostics --config configs/diagnostics/fastapi.yaml'
make cli ARGS='diagnose --config configs/diagnostics/fastapi.yaml'
```

準備では公式スナップショット2件と、ハッシュで固定したCPU用依存関係を取得します。通常の診断ではネットワークやモデルを使用せず、ケースごとに新しい作業コピーでテストを実行します。原本と既存の評価結果は変更しません。

ローカル診断の完了後、同じ8条件をGPUなしのPrivate Notebookへ渡します。

```sh
make cli ARGS='prepare-kaggle --name diagnostic-cpu-20261020 --diagnostics-config configs/diagnostics/fastapi.yaml'
kaggle kernels push -p artifacts/kaggle/diagnostic-cpu-20261020
```

診断結果は`artifacts/diagnostics/`に保存し、公式のエージェント評価と分けます。診断パッチのテスト通過は、エージェントが課題を解決した実績へ加算しません。詳しくは[パッチ診断](docs/diagnostics.md)を参照してください。

## 提出前の検証と正式提出

開発環境と診断環境の準備後、現在のエージェントを一度だけZIPへ固定します。

```sh
make setup
make cli ARGS='prepare-diagnostics --config configs/diagnostics/fastapi.yaml'
make cli ARGS='preflight --config configs/preflight.yaml'
```

事前検証は品質・構造・パッチ診断・提出設定の参照・ZIPの展開前後の一致を確認します。通常の実行はオフラインです。入力が不足する場合は準備段階へ戻り、自動取得しません。`configs/preflight.yaml`には現在のエージェント、評価設定、保存済みZIPから再利用するLoRAの取得元とSHA-256を指定します。取得元ZIPは既存の公式評価`20261020T072100Z-e679cea7`の保存結果です。新しい環境では先にその入力を取得し、設定にあるSHA-256と照合してください。

表示された検証記録を`REPORT`へ指定し、CPU検証とGPU事前評価に同じ候補を渡します。

```sh
REPORT=artifacts/preflight/実行ID/report.json
make cli ARGS="prepare-kaggle --name candidate-cpu --preflight-report $REPORT --verify-only"
kaggle kernels push -p artifacts/kaggle/candidate-cpu
```

生成先の名前は毎回新しく指定します。既存フォルダーは上書きしません。CPUの公式コンパイルと2課題の参照検証を通過後、同じ記録からGPU用コードを生成します。

```sh
make cli ARGS="prepare-kaggle --name candidate-gpu --preflight-report $REPORT"
kaggle kernels push -p artifacts/kaggle/candidate-gpu
kaggle kernels status satter0927/gemma-4-developer-agent-baseline
kaggle kernels logs satter0927/gemma-4-developer-agent-baseline --follow
```

正常終了後、結果を新しい保存先へ取得します。展開した提出物や一時キャッシュを除き、正式提出用ZIPと検証記録を保存します。

```sh
kaggle kernels output satter0927/gemma-4-developer-agent-baseline -p artifacts/results/candidate-gpu --page-size 200 --file-pattern '^(submission.zip|artifacts/results/[^/]+/(?!submission/|wheel-links/).*)$'
```

Notebookは非公開で、インターネット接続は無効です。CPUではモデル起動を行わず、GPUではKaggle側のL4を使用します。MacのGPUは使いません。GPU事前評価の対象は2課題です。課題ごとの公式判定、パッチ、ログ、トレース、集計と終了処理を確認し、解決数0件でも正常完了と性能上の失敗を区別します。

CPU・GPUのソースSHA-256、候補ZIP、依存関係、課題データ、評価設定を照合します。Kaggle上でサンプルからZIPを作り直しません。入力・証拠が変わった場合は事前検証からやり直します。Notebook実行は正式提出とは別です。

全検証の正常終了後だけ`/kaggle/working/submission.zip`が出力されます。取得したZIPのSHA-256が検証記録と一致すること、Kaggleの参加状態と提出可能枠を確認して、同じファイルを提出します。

```sh
kaggle competitions submit gemma-4-developer-agent -f artifacts/results/candidate-gpu/submission.zip -m 'Initial validated agent'
kaggle competitions submissions gemma-4-developer-agent --format json
```

提出ID・日時・ZIPのハッシュ・採点状態・公開スコアを記録します。受付が不明なときは提出一覧を確認し、同じ操作を繰り返しません。非公開課題の本番スコアは正式提出後に分かります。開発用Notebookのsubprocess環境と本番のDocker環境には差があるため、2課題の成功を本番全体の保証にしません。

## 開発規約

[コーディング規約](docs/coding-standards.md)に設定・定数・依存・結果保存の方針をまとめています。構造チェックは循環依存、依存方向、未分類モジュール、運用パラメータの直書きを検出します。`make verify`は準備済み環境のチェックだけを実行し、`make check`は開発環境を更新してから同じチェックを実行します。

## 結果を確認する

```sh
make cli ARGS='results'
make cli ARGS='show RUN_ID'
make cli ARGS='show RUN_ID --task fastapi_15661 --artifact patch'
make cli ARGS='compare RUN_ID_A RUN_ID_B'
```

`RUN_ID`には`results`で表示された実行IDを指定します。詳細ファイルは`patch`、`test-log`、`agent-log`、`trace`から選べます。元のログとトレースは保存時の内容で表示します。

一覧には状態、最後の実行段階、予定・完了・解決件数、解決率、所要時間、実行種別を表示します。参照検証と未完了評価の解決率は表示しません。参照検証はエージェント評価と比較できません。評価開始前に停止した実行は、未生成の評価ファイルを破損扱いせず、失敗状態を表示します。比較では課題ごとの結果と時間、実行条件の差を確認できます。プロンプト・エージェント設定・LoRAを含む提出ファイルの変更も表示します。正常な表示と結果未取得は終了コード0、読み込み・検証エラーは1、引数エラーは2です。

## 構成

- `agent/`：提出するYAMLとプロンプト。配布LoRAは固定した取得元から候補の生成時に追加します。
- `src/gemma_agent_dev/`：設定、パッチ診断、公式基盤との接続、Kaggle実行、結果処理、CLI。
- `configs/`、`kaggle/`：実行条件とKaggle起動処理。
- `tests/`、`scripts/`、`docs/`：テスト、品質チェック用スクリプト、設計、運用手順、検証状況。
- `artifacts/`：Kaggle用コードは`kaggle/`、公式評価は`results/`、診断は`diagnostics/`、参照入力は`reference/`へ保存します。開発環境は`venv/`、課題用環境は`task-envs/`、キャッシュと一時ファイルは`cache/`と`tmp/`に置きます。すべてGit管理対象外です。

初回提出候補のCPU検証とGPU事前評価は正常終了し、2課題の解決数は0/2でした。検証した同じZIPを提出ID`57441297`として正式提出しています。公開スコアは採点待ちです。詳しい条件と記録は検証状況にまとめています。

詳しくは[設計](docs/design.md)、[評価と提出の手順](docs/evaluation.md)、[検証状況](docs/validation.md)を参照してください。

GitHubリポジトリはPublic、開発用Kaggle NotebookはPrivateです。競技データ、パッチ、LoRA、実行ログ、認証情報はGitへ保存しません。
