# Gemma 4 Developer Agent

KaggleのGemma 4 Developer Agent Competition向けの提出エージェントと開発用CLIです。保存パッチの診断、固定課題とEasy課題群の比較、候補の検証、期限付きの提出管理を分けて行います。

| 場所 | 内容 |
|---|---|
| `agent/` | 正式提出するエージェントの宣言・指示・ツール |
| `src/gemma_agent_dev/` | 診断・評価・候補・Notebook・提出の開発用コード |
| `configs/` | 課題、評価、診断、環境、提出の設定 |
| `docs/` | 分析、設計、操作、参照、実験記録 |
| `artifacts/` | Git管理外の入力、環境、ZIP、実行結果 |

[文書案内](docs/README.md)から、[競技と採点](docs/analysis/competition.md)、[評価実績](docs/analysis/tasks-and-results.md)、[構造](docs/design/architecture.md)、[提出手順](docs/operations/submission.md)を確認できます。Easy課題の改善とMLflowによる記録管理は[実験手順](docs/operations/experiments.md)を参照してください。過去の条件と操作は[記録](docs/records/README.md)へ保持しています。

## 最初の準備

Python 3.12とuvを用意し、リポジトリ内で実行します。

```sh
make setup
make check
make cli ARGS='--help'
```

競技データと準備済みCPU環境を使う操作は[環境の準備](docs/operations/setup.md)、課題の再検証は[パッチ診断](docs/operations/diagnostics.md)、候補の固定と結果確認は[評価手順](docs/operations/evaluation.md)を参照してください。開発時は[コーディング規約](docs/coding-standards.md)に従います。

通常の品質検査と事前検証はKaggle実行・正式提出を開始しません。定期提出の有効化と認証設定は[提出手順](docs/operations/submission.md)で明示的に行います。

日常の比較準備・パッチ診断・結果整理・候補準備は[開発フロー](docs/operations/workflows.md)を参照してください。開発AI用の`gemma-dev-workflows` Skillも同じCLIを使用します。
