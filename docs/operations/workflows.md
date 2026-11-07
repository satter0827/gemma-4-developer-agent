# 日常の開発フロー

準備、保存パッチ診断、結果整理、候補登録準備を目的別に実行します。成果物と次の操作はJSONで取得できます。GPU起動、状態への登録、正式提出は別の明示操作です。

最初に`make setup`で開発用パッケージを更新します。ソース変更後の品質検査は`make check`を使います。候補固定にも品質検査が含まれるため、同じワークフロー内で重ねて実行する必要はありません。

| 操作 | 完了すること | 完了しないこと |
|---|---|---|
| `prepare` | 候補固定と同一バンドルのCPU検証 | モデル生成、GPU評価、精度判定 |
| `diagnose` | 保存パッチと対照の再現 | 新しい回答の生成、過去の成績更新 |
| `report` | 保存結果の照合と分析 | 自動採用、登録、提出 |
| `candidate` | 保存版ZIPまで照合した候補記録の作成 | 提出状態の変更、正式提出 |

## 比較の事前CPU検証

```sh
make cli ARGS='workflow prepare --config configs/evaluation/easy-completion-v1.yaml --sdk-config configs/environments/sdk-cpu.yaml --json'
```

入力確認、必要な課題・SDK環境の準備、品質検査、対照検証、候補固定、CPUバンドル生成、固定SDKの検証を順に行います。課題環境とSDK環境の依存は混ぜません。準備済み環境は入力と実際の依存を照合します。改変されたソースや依存を自動で正しい証拠へ置き換えません。

`stage=cpu_verified`は事前CPU検証の完了です。`completion_budget`はGPUジョブ全体の上限で、実測時間ではありません。準備上限が不明、全課題が期限に収まらない、quota不足の場合はGPU用生成を拒否します。従来の90分設定で全6組を実行する条件は拒否し、単一担当の比較には明示した177分の設定を使います。

GPU実行物を生成するときは返された`freeze`と`cpu_verification`のパスを既存の`prepare-kaggle`へ渡し、最新quotaを明示します。起動の確認と操作は[実験手順](experiments.md)を参照します。

## 保存パッチの診断

```sh
make cli ARGS='workflow diagnose --config configs/diagnostics/fastapi.yaml --json'
```

`stage=patch_diagnosed`は参照、未修正、生成、診断パッチの期待結果が一致したことを示します。失敗の再現が期待結果なら、テスト不合格でも診断は成立します。テスト0件、収集失敗、未完了は通常の通過と区別します。詳細は[パッチ診断](diagnostics.md)を参照します。

## 結果の整理

```sh
make cli ARGS='workflow report --input artifacts/results/RUN/session.json --json'
make cli ARGS='workflow report --input artifacts/results/RUN/manifest.json --json'
```

対比較と単独評価を記録の種別で判別し、それぞれの照合・分析処理を使います。未完了・未実行を失敗へ変換せず、Notebook評価と正式スコアを分けます。

遠隔の結果は、保存版、SDK環境のPython、提出設定、結果manifestのリモートパスを指定して回収します。

```sh
make cli ARGS='workflow report --remote-kernel OWNER/KERNEL --remote-version VERSION --remote-path artifacts/results/RUN/session.json --policy configs/submission/policy-read-v2.yaml --api-python artifacts/venv/bin/python --json'
```

版指定のファイル一覧とダウンロードを使用し、取得上限とページを確認します。一時領域で証拠を検証してから`artifacts/imports/`へ保存します。状態表示だけでは対象版の結果と判断しません。

SDK環境の`bin/python`をそのまま指定します。読み取り設定`policy-read-v2.yaml`では、一時的なHTTP 429・5xxに対し、初回を含め最大5回まで再試行します。待ち時間はサーバーの指示と残時間を照合し、1回のSDK呼び出し全体の期限を維持します。結果全体の回収時間とは区別します。アップロードと提出送信は1回だけ実行します。

公開記録は、対比較へ`--review REVIEW.yaml --record-output docs/records/experiments/NEW_ID.json`を追加して作成します。記入項目は[実験手順](experiments.md)を参照します。`--tracking-config configs/environments/mlflow.yaml`を追加した場合だけ、準備済みMLflow環境へ同期します。公開記録は上書きしません。

## 候補登録の準備

`SOURCE.json`には保存版の`owner`、`kernel`、`version`、ZIPの`path`を指定します。

```sh
make cli ARGS='workflow candidate --cpu CPU_SESSION --gpu GPU_SESSION --variant VARIANT --source SOURCE.json --policy configs/submission/policy-read-v2.yaml --api-python artifacts/venv/bin/python --json'
```

同じ定義・候補・入力・SDKのCPU/GPU証拠、全課題の完了、保存版から取得した同一ZIPを照合します。候補記録は非公開のワークフロー保存先へ作成します。登録と提出は[提出手順](submission.md)の明示操作を使います。

## 結果と再開

JSONは`workflow_id`、`status`、`stage`、`artifacts`、`reason_codes`、`next_actions`を持ちます。`completion`で品質検査、CPU検証、GPU評価、比較、正式提出の完了・未完了・未実行を分けます。成功は終了コード0、実行失敗は1、引数の誤りは2です。進捗は標準エラーへ出力します。

```sh
make cli ARGS='workflow status --run WORKFLOW_JSON --json'
make cli ARGS='workflow prepare --config EXPERIMENT.yaml --sdk-config SDK.yaml --resume WORKFLOW_JSON --json'
make cli ARGS='workflow diagnose --config DIAGNOSTICS.yaml --resume WORKFLOW_JSON --json'
```

完了工程は入力と保存ファイルを再照合して省略し、未完了工程は新しい保存先で実行します。入力変更や証拠改変があれば停止します。過去の部分結果へ継ぎ足したり、GPU内の会話や評価途中の状態を再開したりしません。

同じチェックアウトでワークフローを同時実行できません。ロックはプロセス終了時に解放されます。保存記録は参照と工程結果だけを持ち、診断・CPU/GPUのmanifestが証拠の正本です。

## 品質記録を再利用する

`workflow prepare --quality-report artifacts/.../quality.json`で、完了した品質検査を明示的に再利用できます。`make check`、`make setup-submission`、`make verify`の順序、終了コード、ログのハッシュ、ソース・テスト・提出資産・設定・文書のハッシュを照合します。入力や記録が変わった場合は停止します。過去の記録を更新せず、新しい出力先で検査し直してください。

## 単一担当の比較を実行する

`easy-focused-v1`は、`workflow-v1`と`workflow-focused-v1`の提出構成全体を比較します。単一担当、Skillによる作業手順、出力上限の変更を含むため、個別変更の寄与は判定できません。`easy-focused-confirmation-v1`は、事前に固定した未使用6問による独立した比較です。

両実験を`workflow prepare`で準備し、候補のZIPハッシュが一致することを確認してからGPU実行物を作成します。実行前後のquotaを保存し、追加予算13時間、予備1時間、最大2回を通して照合します。各実行の177分は、準備20分、モデル起動20分、12評価それぞれ11分、後始末5分から計算します。評価期限にはsandbox準備、生成、採点、保存が含まれます。従来の評価追加時間の見積もりとは区別します。

調整用比較が全件有効に完了し、解決数が純増した場合だけ独立比較へ進みます。独立比較でも純増した場合に採用を推薦します。同点、減少、修正消失、検証の誤成功、証拠不整合、基盤側の未完了があれば採用を保留します。新たに失敗した課題も結果へ含めます。途中の会話や評価は再開せず、課題の差し替え、予算の延長、GPUの自動再試行は行いません。

GPU評価から回収した保存版ZIPをそのまま提出候補として保持します。候補準備、候補登録、正式提出は別の操作です。

`workflow-focused-v2`は、実装と同じディレクトリにあるテストだけの変更を除外する修正版です。pytestが実際に収集したパスと命名設定を使い、CPU検証も旧版の誤成功を拒否します。`easy-focused-verification-v2`でCPU検証し、評価済みの旧版ZIPと分けて保持します。修正版のGPU成績は未確認です。

品質記録を新たに作る場合は、準備済み開発環境から次を実行します。検査中は同じチェックアウトを変更せず、候補準備と並行実行しないでください。

```python
from pathlib import Path

import yaml

from gemma_agent_dev.candidates.quality import run
from gemma_agent_dev.common.artifacts import new_run_id
from gemma_agent_dev.workflows.records import exclusive

root = Path.cwd()
settings = yaml.safe_load((root / "configs/evaluation/preflight-v2.yaml").read_text())
with exclusive(root):
    report = run(root, root / "artifacts/quality" / new_run_id(), settings["quality_timeout_seconds"])
print(report)
```

Notebookの評価結果は、ワーカーと所有プロセスの後始末が終わってから完了として扱います。結果回収は同じNotebook版の監視記録とプロセス記録も取得し、実行ID・ソース・ハッシュを照合します。評価済みの課題があっても、後始末や記録保存が未完了なら候補登録へ進めません。

ローカルCPU検証は、凍結したソースZIPから直接実行します。GPU用Notebookの埋め込みサイズ上限は適用しません。`workflow prepare`の`source_bundle`をGPU実行物へ渡します。Kaggleのソース上限を超える場合は、そのZIPと同じバイト列を`source-bundle.bin`としてPrivate Datasetへ置き、`prepare-kaggle --bundle-dataset owner/name`で参照します。DatasetへのアップロードとGPU起動は、それぞれ既存の明示操作で行います。
