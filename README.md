# Gemma 4 Developer Agent

KaggleのGemma 4 Developer Agent Competitionに向けた、提出エージェントと評価用CLIです。評価対象、設定、成果物を固定し、環境の動作とエージェントの解決能力を分けて確認します。

全129問の分類と費用の読み方は[問題の分析と管理](docs/analysis/README.md)、検証済み候補の朝の自動提出と通知設定は[提出手順](docs/submission.md)を参照してください。定期提出はmain反映・認証設定・有効化が完了してから動きます。

## 現在の評価

初回の正式提出IDは`57441297`、公開スコアは**0.01**です。同じZIPによる開発用評価はFastAPIの2課題で**0/2**でした。正式評価と開発用評価では対象課題が異なるため、この数値は矛盾しません。正式評価の課題別ログがないため、低スコアの原因を時間切れだけとは断定しません。

2026年10月21日の別の開発用比較では、同じ6課題で現行指示が**2/6**、改善指示が**3/6**でした。主エージェントの指示だけを変え、実行コード・依存関係・設定・課題順を照合しています。生成結果12件の成否はMacでも再現しました。6課題の単回比較であり、正式評価への効果は未確認です。今回のGPU消費は2.83030 quota時間で、正式な再提出はしていません。

診断ケースの一致数、プロジェクトのテスト件数・カバレッジ、エージェントの解決数、正式スコアは別の指標です。[採点差と公開情報](docs/research.md)と[検証状況](docs/validation.md)に証拠と限界を記載します。

## 構成

| 場所 | 役割 |
|---|---|
| `agent/` | 共通のモデル・LoRA・ツール宣言、生成設定、比較する主エージェントの指示 |
| `configs/` | 共通課題定義、診断、実行プロファイル、CPU SDK環境、GPU予算 |
| `src/gemma_agent_dev/` | 設定、診断、公式接続、Kaggle実行、結果分析、CLI |
| `kaggle/` | バンドルの照合・展開と実行期限の適用 |
| `tests/`、`scripts/` | 動作・構造・カバレッジの検証 |
| `docs/` | 設計、運用、公開情報と検証記録 |
| `artifacts/` | 参照入力、専用環境、診断、評価、分析、ログ。Git管理対象外 |

## 準備

Python 3.12、uv、Gitを使用します。公式入力の取得とKaggle実行には、認証済みKaggle CLIとコンペ参加が必要です。コマンドはリポジトリのルートで実行します。

```sh
make setup
make verify
make cli ARGS='prepare-diagnostics --config configs/diagnostics/fastapi.yaml'
make cli ARGS='prepare-diagnostics --config configs/diagnostics/comparison.yaml'
make cli ARGS='prepare-runtime-checks --config configs/runtime-checks.yaml'
```

`make setup`は開発環境、`prepare-diagnostics`は課題のスナップショットとハッシュで固定したCPU依存、`prepare-runtime-checks`は公式SDKの専用CPU環境を準備します。開発環境と課題環境を共有せず、MacへLinux・CUDA用Wheelを導入しません。

準備だけが取得・依存更新を行います。通常の診断と事前検証は、準備済みの入力を使用します。入力の不足や変更を検出したら停止します。

## ローカル診断

```sh
make cli ARGS='diagnose --config configs/diagnostics/fastapi.yaml'
make cli ARGS='qualify --config configs/comparison.yaml'
make cli ARGS='check-runtime --config configs/runtime-checks.yaml'
```

既存2課題は、未修正・参照パッチ・保存された生成パッチ・原因を一つ変えた診断パッチの8条件で再現します。診断資産はエージェントへ渡しません。

比較課題はFastAPI・Requests・Richから2件ずつ、事前に決めた順序で選びます。実際のテストプロセスが作業コピーを読み、未修正が失敗し参照パッチが通ることが採用条件です。補欠と除外理由を記録し、モデル実行前に`artifacts/qualification/qualified.json`へ固定します。固定後の再選定は拒否します。6課題が揃わなければGPU評価を行いません。

`check-runtime`は合成入力でコンパイル、モデル・LoRA登録、生成リクエストと思考履歴を確認します。重みの読み込みや推論の検証はGPU側で行います。

## 候補の固定とPrivate CPU

```sh
make cli ARGS='preflight --config configs/preflight.yaml --profile baseline'
make cli ARGS='preflight --config configs/preflight.yaml --profile workflow'
```

プロファイルの指定は必須です。`smoke`は1分・10ツールの動作確認、`baseline`と`workflow`は5分・30ツールの比較です。比較では主エージェントの指示だけを変え、モデル、LoRA、補助エージェント、ツール、生成設定を揃えます。thinking budgetは4096、seedは0です。seedだけで完全な再現性は保証しません。

事前検証は品質チェックと既存8条件の診断を実行し、現在のエージェントと固定済み配布LoRAをZIPへ保存します。表示された検証記録を使います。

```sh
REPORT=artifacts/preflight/実行ID/report.json
make cli ARGS="prepare-kaggle --name baseline-cpu --preflight-report $REPORT --verify-only"
kaggle kernels push -p artifacts/kaggle/baseline-cpu
kaggle kernels status satter0927/gemma-4-developer-agent-verification
```

NotebookはPrivate、インターネット無効、GPUなしです。公式コンパイルと同じ6課題の参照検証を行います。各候補で実行し、結果は別々の保存先へ取得します。既存8条件のCPU診断は`prepare-kaggle --name 名前 --diagnostics-config configs/diagnostics/fastapi.yaml`で別に生成できます。

```sh
kaggle kernels output satter0927/gemma-4-developer-agent-verification -p artifacts/results/baseline-cpu --page-size 200 --file-pattern '^(artifacts/results/[^/]+/(?!submission/|wheel-links/).*|artifacts/jobs/.*)$'
```

## Private GPU比較

CPU結果の`manifest.json`を指定します。候補ZIP、実行ソース、課題順、公式Wheel、読み込み元の証拠が一致しなければGPU用コードを生成しません。

```sh
CPU=artifacts/results/baseline-cpu/artifacts/results/実行ID/manifest.json
make cli ARGS="prepare-kaggle --name baseline-gpu --preflight-report $REPORT --cpu-report $CPU"
kaggle quota --format json
kaggle kernels push -p artifacts/kaggle/baseline-gpu
```

GPUはKaggleのL4を使用し、ローカルGPUは使いません。各Notebookは準備から保存・終了まで45分を上限に、監視プロセスが子プロセス群を停止します。結果と監視記録を取得します。

```sh
kaggle kernels output satter0927/gemma-4-developer-agent-baseline -p artifacts/results/baseline-gpu --page-size 200 --file-pattern '^(artifacts/results/[^/]+/(?!submission/|wheel-links/).*|artifacts/jobs/.*)$'
```

実行前に、公式APIと今回の実行記録から得た消費量・残枠・完了回数で判定します。下記の変数には観測値を指定します。

```sh
make cli ARGS="check-gpu-budget --config configs/gpu-budget.yaml --consumed-hours $CONSUMED_HOURS --remaining-hours $REMAINING_HOURS --completed-runs $COMPLETED_RUNS"
```

別の実行が動いている場合は`--active`を付けます。情報が不明なら実行せず、実行不可の判定は終了コード1になります。

今回のGPU予算は最大4 quota時間分、そのうち1時間分を予備に確保します。`configs/gpu-budget.yaml`で管理し、実行前後の公式残枠を保存します。累積消費と次回最大消費と予備が上限を超える場合、残枠が不明な場合、別の実行が動いている場合は開始しません。自動再試行と同時実行は行いません。回復日時は公式の`kaggle quota`で確認します。

現行指示と改善指示を同じ6課題へ順番に実行します。追加測定は予算内で最大1回です。改善が見えたら改善条件、それ以外は現行条件を6課題すべて再測定します。少数の公開課題の成績を正式評価の予測や改善の確定には使いません。

この経路は正式提出用の`/kaggle/working/submission.zip`を出力・送信しません。候補ZIPは実行結果内に保持します。正式提出とPRの承認・マージは別の操作です。

## 結果分析

```sh
make cli ARGS='results'
make cli ARGS='show RUN_ID'
make cli ARGS='show RUN_ID --task 課題ID --artifact trace'
make cli ARGS='show RUN_ID --task 課題ID --artifact patch'
make cli ARGS='compare RUN_ID_A RUN_ID_B'
make cli ARGS='analyze RUN_ID_A --other RUN_ID_B'
```

分析は`artifacts/analysis/`へ保存し、公式の結果を変更しません。課題単位の公式判定、採点ログの最終pytest集計、最初の編集試行、編集失敗、検証コマンド、パッチ保存履歴、制限の証拠を記録します。公式のツール消費回数とトレース上の全回数を分け、最終パッチとログもハッシュで参照できます。収集エラーを実行済みテストへ数えず、設定にあるタイムアウト値を実際の時間切れとは判定しません。検証コマンドを実行しただけでは解決と判定せず、取得できない情報は不明として扱います。

## 品質と開発規約

```sh
make check
make coverage
```

formatter、lint、型チェック、構造テスト、動作テストを実行します。行90%以上・分岐80%以上を別々に必須とし、`artifacts/coverage/`にJSON、HTML、判定表を保存します。品質検査は外部課題・評価SDK・認証情報・GPUを使いません。CIの提出SDK契約テストは固定版Kaggle SDKと合成HTTP応答を使い、外部通信を行いません。

[コーディング規約](docs/coding-standards.md)では設定値と定数、依存方向、循環参照の禁止、副作用と結果保存を定めます。[設計](docs/design.md)、[実行手順](docs/evaluation.md)、[パッチ診断](docs/diagnostics.md)も参照してください。競技データ、正解・診断パッチ、LoRA、実行ログ、認証情報はGitへ含めません。
