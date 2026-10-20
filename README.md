# Gemma 4 Developer Agent

KaggleのGemma 4 Developer Agent Competitionに向けた、提出エージェントと評価用CLIです。公式サンプルの構成を維持し、評価条件と提出物を追跡します。

## 開発環境

Python 3.12、uv、Gitを使用します。Kaggleでの実行には、認証済みのKaggle CLIとコンペ参加が必要です。

```sh
cd /Users/takafumi/Developer/personal/gemma-4-developer-agent
make setup
make check
```

開発環境はiCloud外に置きます。Makefileを通して実行すると、仮想環境、キャッシュ、一時ファイルはすべて`artifacts/`へ保存されます。通常のwheelとしてインストールし、コード変更後は`make setup`で更新します。`make check`は更新とすべての品質チェックを実行します。

## Kaggleで評価する

先にGPUを使わない参照検証を実行します。

```sh
make cli ARGS='prepare-kaggle --name verify-20261020 --verify-only'
kaggle kernels push -p artifacts/kaggle/verify-20261020
kaggle kernels status satter0927/gemma-4-developer-agent-verification
kaggle kernels logs satter0927/gemma-4-developer-agent-verification --follow
```

`fastapi_15661`・`fastapi_15588`の両方で、修正前が未解決、参照パッチ適用後が解決になることを確認します。通過後、同じソースでGPU評価を実行します。

```sh
make cli ARGS='prepare-kaggle --name baseline-20261020'
kaggle kernels push -p artifacts/kaggle/baseline-20261020
kaggle kernels status satter0927/gemma-4-developer-agent-baseline
kaggle kernels logs satter0927/gemma-4-developer-agent-baseline --follow
kaggle quota
kaggle kernels output satter0927/gemma-4-developer-agent-baseline -p artifacts/results/baseline-20261020 --page-size 200 --file-pattern '^artifacts/results/[^/]+/(?!submission/|wheel-links/).*'
```

生成するNotebookはPrivate、インターネット接続は無効です。事前検証ではモデル重みを入力に付けず、推論サーバーを起動しません。GPU評価でも同じ2課題の参照検証をやり直し、通過後にモデルを起動します。HTTPX・Requestsを含む全公開課題の環境が正常になったことは保証しません。

生成先の名前は毎回新しく指定してください。既存フォルダーへの上書きはできません。送信対象とソースのハッシュは`source-manifest.json`で確認できます。2種類の実行は同じソースZIPを使い、実行種別は起動時に渡します。

時間制限を比較するときは、両方の生成コマンドに同じ評価設定を指定します。省略時は1分設定のままです。

```sh
make cli ARGS='prepare-kaggle --name verify-5min --verify-only --eval-config configs/evaluation/5min.yaml'
make cli ARGS='prepare-kaggle --name evaluation-5min --eval-config configs/evaluation/5min.yaml'
```

評価設定は、プロジェクト直下からの相対パス、またはプロジェクト内の絶対パスで指定します。元のファイルは変更せず、選択した設定をソースZIP内の`agent/eval_config.yaml`へ格納します。選択元、SHA-256、適用値は`source-manifest.json`と実行manifestに記録します。

既存実行の状態を確認してから次をアップロードします。コードの保存、実行開始、完了は別々に確認してください。Notebookの実行はコンペへの提出ではありません。

## 結果を確認する

```sh
make cli ARGS='results'
make cli ARGS='show RUN_ID'
make cli ARGS='show RUN_ID --task fastapi_15661 --artifact patch'
make cli ARGS='compare RUN_ID_A RUN_ID_B'
```

`RUN_ID`には`results`で表示された実行IDを指定します。詳細ファイルは`patch`、`test-log`、`agent-log`、`trace`から選べます。元のログとトレースは保存時の内容で表示します。

一覧には状態、最後の実行段階、予定・完了・解決件数、解決率、所要時間、実行種別を表示します。参照検証と未完了評価の解決率は表示しません。参照検証はエージェント評価と比較できません。評価開始前に停止した実行は、未生成の評価ファイルを破損扱いせず、失敗状態を表示します。比較では課題ごとの結果と時間、実行条件の差を確認できます。正常な表示と結果未取得は終了コード0、読み込み・検証エラーは1、引数エラーは2です。

## 構成

- `agent/`：提出するYAMLとプロンプト。配布LoRAはKaggle上で追加します。
- `src/gemma_agent_dev/`：評価の接続処理、提出物生成、結果の読み込み、CLI。
- `configs/`、`kaggle/`：実行条件とKaggle起動処理。
- `tests/`、`docs/`：テスト、設計、運用手順、検証状況。
- `artifacts/`：`kaggle/`、`results/`、`reference/`、`venv/`、`cache/`、`tmp/`に生成物を集約します。Git管理対象外です。

詳しくは[設計](docs/design.md)、[評価と提出の手順](docs/evaluation.md)、[検証状況](docs/validation.md)を参照してください。

リポジトリと開発用Kaggle資産はコンペ期間中Privateで運用します。終了後の公開は、競技ルールと各資産の再配布条件を確認してから行います。
