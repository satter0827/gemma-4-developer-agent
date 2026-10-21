# 配置変更の検証

確認日：2026年10月21日。PR #5取り込み済みmain `798c6fd`を基点に、コード・設定・文書・テストを責務別に整理しました。主・補助エージェント、プロンプト、モデル、依存バージョン、Wheelハッシュ、実行予算は変更していません。

## 品質検査

| 検査 | 結果 |
|---|---|
| `make -j4 check` | formatter・lint・型・依存・副作用検査を通過、380件成功・提出SDK未導入の2件スキップ |
| `make setup-submission`後の`make verify` | 固定版SDKの合成HTTP契約を含め382件すべて成功 |
| カバレッジ | 行92.43%・分岐83.67%、基準は行90%・分岐80% |
| 文書と差分 | 文書内の相対リンクがすべて解決、`git diff --check`通過 |

新しいKaggle実行、GPU生成、正式提出、定期提出の有効化は行っていません。

## 構造と検証入力

実行コードは`common`、`tasks`、`diagnostics`、`evaluation`、`candidates`、`notebooks`、`submission`へ分割しました。ルートCLIは振り分け、機能別CLIはコマンド定義と表示を担当します。パッケージ初期化に再エクスポートや実行処理はありません。旧importの互換層は設けていません。

設定は`tasks`、`evaluation`、`diagnostics`、`environments`、`submission`へ移しました。Kaggleの資源・マウント先、推論条件、SDK固定情報、課題順を別に読み、バンドルには解決済みの単一設定を保存します。

事前検証では、選択プロファイルと実際の参照入力だけを設定の照合対象にしました。提出方針、手動分類、未選択プロファイルの変更では失効せず、選択された設定やコード・テスト・ビルド・依存定義の変更では失効することを確認しています。Python・依存・Wheel・スナップショットが同じなら、推論条件だけの変更で環境を再作成しません。

インストール照合とimport副作用検査は再帰走査へ変更しました。入れ子の変更・欠落・余分なコード、バンドル内の内部importと親パッケージの欠落を検査します。評価用バンドルには診断パッチと診断実行コードを含めません。起動コードはハッシュ対象のバンドルから読み、後からチェックアウトが変わっても別の起動コードへ差し替わらないことを確認しています。

## ZIPと過去結果の保持

保存済み成果物2,158ファイルのハッシュは変わっていません。固定6問と順序、既存12結果の2/6・3/6、登録候補の取得先・内容ID・検証証拠も保持しています。過去CPU/GPU記録からworkflow候補を再照合できました。

既存の候補生成には、更新時刻でZIP全体のハッシュが変わる問題がありました。候補と実行バンドルのZIP保存処理を共通化し、順序・時刻・圧縮方式を固定しました。同じ入力の再生成は同じバイト列になります。過去のZIPは再包装せず、そのままのハッシュを使用します。

| 候補 | 保存済みZIPのSHA-256 | 同じ内容を正規化した新ZIPのSHA-256 |
|---|---|---|
| baseline | `242f2e87897b31ce0440d2259e5a1e90e7fc4043f5101dfd78bfe5dfc6e36c61` | `d6e06d02b6450a293c2d065682cedf0371a56ea1fcbd9eaa7f7a2e388e7016f8` |
| workflow | `4c49619decaa48f56afe5079c61e0a17dd6f837e23ee441a8069d2d0109deefd` | `573b7c894ca02feebf9c0a877120b2a4ef4ad21db7624de4a2f24e0c3893e845` |

新ZIPの内容一覧は各保存済み候補と一致しています。この再生成を新しいCPU/GPU評価や提出候補登録には使っていません。

## 準備済みCPU環境

| 検証 | 結果 | 新規記録 |
|---|---|---|
| 保存パッチの診断 | 8条件すべて期待結果と一致 | `artifacts/diagnostics/20261021T060451Z-b2351e08/report.json` |
| 固定6問の再検証 | 参照6件が成功、未修正6件が失敗、読み込み元が有効 | `artifacts/qualification/20261021T060536Z-d5b4e26f/report.json` |
| 実SDKのコンパイル・生成リクエスト変換 | 完了、GPUモジュールのimportなし | `artifacts/runtime-checks/20261021T061758Z-45e8fa44/report.json` |
| 台帳 | 129問が一意、履歴なしで生成可能、履歴結合は12結果、再生成が一致 | `artifacts/validation/refactor-20261021/ledger-observations.json` |

固定6問は`fastapi_14786`、`fastapi_14360`、`requests_7315`、`requests_7309`、`rich_3052`、`rich_3180`の順です。再検証は補欠へ入れ替えず、旧`qualified.json`とケースを保持したまま`current.json`へ新しい記録を保存しました。旧環境の準備証拠と直近成功ポインターは検証用領域へ退避しています。

## 提出の未送信確認

固定版Kaggle SDKで履歴・残枠・提出ZIP・Notebook保存版ZIPを実取得しました。workflow候補はNotebook版10から取得したZIPの全体ハッシュと内容一覧が登録情報に一致しました。履歴は1件で、取得前後のIDが一致しています。通信境界で`upload`・`create`を拒否し、状態保存も禁止した状態で検証しました。

提出状態ブランチは未初期化でした。通常の`submission plan`が初期化を要求して終了することと、登録JSONをメモリ上へ読んだ同じCLIの未送信確認が成功することを確認しています。時刻は送信時間外で、公開スコア上位の整理は提出ID`57441297`・0.01、最終選択は未確認です。状態の初期化・候補登録・送信・定期提出の有効化は行っていません。

実接続と保持の証拠は`artifacts/validation/refactor-20261021/readonly-proof.json`、`preservation-proof.json`に保存しました。ZIP、課題データ、パッチ、ログ、認証情報はGitやActions成果物へ含めません。

## 開発入口と移行

AGENTS.mdは規約・設計・操作への入口と実行境界へ絞りました。Makefileは環境更新と準備済み検証を分け、`check`が更新完了後に`verify`を呼ぶ構造にしました。並列指定でも同じ一時領域のテストを重ねません。`make`はヘルプ、`format`は整形、`setup-submission`は提出SDK込みの準備です。

既存環境の移行は次の順で行います。

1. `make setup-submission`で新配置をインストールし、`make verify`で確認します。
2. 診断と比較の`prepare-diagnostics`、`prepare-runtime-checks`を新設定パスで実行します。入力が取得済みならオフラインキャッシュを利用できます。
3. `diagnose`、`qualify --config configs/tasks/comparison.yaml --fixed-selection`、`check-runtime`で新しい証拠を作ります。
4. `task-report --config configs/tasks/report.yaml`で台帳を生成します。過去の成績と費用が必要な場合だけ`--observations-config configs/evaluation/observations.yaml`を指定します。
5. 新しい候補を作る場合は現在の事前検証とCPU証拠を使います。既存登録候補は保存版の証拠で照合し直せるため、再評価や再登録は不要です。

提出の初期化と有効化は[提出手順](../operations/submission.md)に従います。旧設定パスの互換層はありません。過去の記録にある旧パスは、その時点の条件を示しています。
