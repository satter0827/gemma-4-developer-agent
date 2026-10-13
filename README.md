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

```sh
make cli ARGS='prepare-kaggle --name baseline-02'
kaggle kernels push -p artifacts/kaggle/baseline-02
kaggle kernels status satter0927/gemma-4-developer-agent-baseline
kaggle kernels logs satter0927/gemma-4-developer-agent-baseline
kaggle kernels files satter0927/gemma-4-developer-agent-baseline
kaggle quota
kaggle kernels output satter0927/gemma-4-developer-agent-baseline -p artifacts/results/baseline-02
```

生成するNotebookはPrivate、インターネット接続は無効です。初回は公式スターターと同じ2課題を評価します。評価前に、各リポジトリの修正前・正解パッチ適用後の検証を行います。前提検証に失敗した場合は推論を開始せず、原因をログに保存します。

生成先の名前は毎回新しく指定してください。既存フォルダーへの上書きはできません。送信対象とソースのハッシュは`source-manifest.json`で確認できます。モデルとデータはKaggleの入力から読み込みます。

既存実行の状態を確認してから次をアップロードします。コードの保存、実行開始、完了は別々に確認してください。Notebookの実行はコンペへの提出ではありません。

## 結果を確認する

```sh
make cli ARGS='results'
make cli ARGS='show RUN_ID'
make cli ARGS='show RUN_ID --task fastapi_15661 --artifact patch'
make cli ARGS='compare RUN_ID_A RUN_ID_B'
```

`RUN_ID`には`results`で表示された実行IDを指定します。詳細ファイルは`patch`、`test-log`、`agent-log`、`trace`から選べます。元のログとトレースは保存時の内容で表示します。

一覧には状態、最後の実行段階、予定・完了・解決件数、解決率、所要時間を表示します。未完了の解決率は表示しません。比較では課題ごとの結果と時間、実行条件の差を確認できます。正常な表示と結果未取得は終了コード0、読み込み・検証エラーは1、引数エラーは2です。

## 構成

- `agent/`：提出するYAMLとプロンプト。配布LoRAはKaggle上で追加します。
- `src/gemma_agent_dev/`：評価の接続処理、提出物生成、結果の読み込み、CLI。
- `configs/`、`kaggle/`：実行条件とKaggle起動処理。
- `tests/`、`docs/`：テスト、設計、運用手順、検証状況。
- `artifacts/`：`kaggle/`、`results/`、`reference/`、`venv/`、`cache/`、`tmp/`に生成物を集約します。Git管理対象外です。

詳しくは[設計](docs/design.md)、[評価と提出の手順](docs/evaluation.md)、[検証状況](docs/validation.md)を参照してください。

リポジトリと開発用Kaggle資産はコンペ期間中Privateで運用します。終了後の公開は、競技ルールと各資産の再配布条件を確認してから行います。
