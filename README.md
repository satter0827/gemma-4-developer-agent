# Gemma 4 Developer Agent

KaggleのGemma 4 Developer Agent Competitionに向けた、提出エージェントと評価結果の閲覧アプリです。公式サンプルの構成を維持し、評価条件と提出物を追跡できるようにします。

## 開発環境

Python 3.12、uv、Gitを使用します。Kaggleでの実行には、認証済みのKaggle CLIとコンペ参加が必要です。

```sh
uv sync --frozen --no-editable --reinstall-package gemma-agent-dev
make check
```

通常のwheelとしてインストールし、テストでも配布されるパッケージを読み込みます。コード変更後は再インストールが必要です。`make check`は再インストールとすべての品質チェックを実行します。

## Kaggleで評価する

```sh
uv run --no-sync gemma-dev prepare-kaggle --output build/kaggle-run-01
kaggle kernels push -p build/kaggle-run-01
kaggle kernels status satter0927/gemma-4-developer-agent-baseline
kaggle kernels output satter0927/gemma-4-developer-agent-baseline -p results/kaggle-run-01
```

生成するNotebookはPrivate、インターネット接続は無効です。初回は公式スターターと同じ2課題を評価します。評価の前に、各リポジトリの修正前・正解パッチ適用後の検証を行います。前提の検証に失敗した場合は推論を開始せず、原因をログに保存します。

出力先は毎回新しく指定してください。アップロードするファイルとソースのハッシュは、生成先の`source-manifest.json`で確認できます。モデルとデータはKaggleの入力から読み込み、ローカルからアップロードしません。

評価が実行されると、結果フォルダーに`manifest.json`、公式の集計・課題別結果・ログと、検証済みの`submission.zip`を保存します。Notebookの実行はコンペへの提出ではありません。

## 結果を見る

```sh
uv run --no-sync gemma-dev view --results results
```

ブラウザーで `http://127.0.0.1:8501` を開きます。結果の閲覧と比較だけを行い、評価の開始やデータの変更は行いません。未完了・破損した実験には解決率を表示せず、異なる実行条件の比較には注意を表示します。

## 構成

- `agent/`：提出するYAMLとプロンプト。配布LoRAはKaggle上で追加します。
- `src/gemma_agent_dev/`：評価の接続処理、提出物生成、結果の読み込み、閲覧画面。
- `configs/`、`kaggle/`：GPU・課題・依存関係の設定とKaggle起動処理。
- `tests/`：設定、提出物、結果の整合性、画面のテスト。
- `docs/`：設計、評価手順、出典、検証状況。

詳しくは[設計](docs/design.md)、[評価と提出の手順](docs/evaluation.md)、[検証状況](docs/validation.md)を参照してください。

リポジトリはコンペ期間中Privateで運用します。終了後の公開は、競技ルールと各資産の再配布条件を確認してから行います。
