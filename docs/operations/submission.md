# 検証済み候補の定期提出

新しく登録した候補だけを、提出枠と時刻を確認して送信します。NotebookのGPU計算は実行せず、保存済みの非公開Notebook版から同一ZIPを取得します。実験の自動昇格は行いません。

## 実行時刻と期限

| 日本時間 | 処理 |
|---|---|
| 08:15 | 履歴と残枠を確認し、登録順に1候補を提出 |
| 08:45 | 当日分の受付が未確認の場合だけ予備実行 |
| 20:15 | 採点結果と公開スコア上位2件を整理 |

設定は`configs/submission/policy.yaml`で管理します。UTCの提出枠と日本時間の送信時刻を分けて判定します。定時処理は08:55以降に正式提出を開始しません。明示した手動提出は朝の時間帯以外も送信でき、競技最終日の08:55を共通の期限とします。アップロード後にも履歴・残枠を取得し、正式受付APIの直前に再度時計を確認します。

競技の締切は2026年12月3日08:59、運用上の送信期限は同日08:55です。締切後は夜の結果確認だけを残し、12月4日20:15の最終確認後にワークフローを無効化します。未完了の採点は最終要約に残します。GitHubの定期実行には遅延・欠落があるため、定刻実行やサーバー側の受付完了時刻は保証しません。[競技日程](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/timeline)、[GitHubの定期実行](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)

## 有効化

1. 実装をmainへ反映します。
2. [KaggleのAPI Tokens設定](https://www.kaggle.com/settings/api)で定期実行用のAPI Tokenを用意し、リポジトリのActions Secretへ`KAGGLE_API_TOKEN`として登録します。値はコード・ログ・状態ブランチへ記録しません。
3. ローカルでは`make setup-submission`で提出専用依存を追加します。別の既存SDK環境を使う場合は、提出コマンドの`--api-python`でPythonを指定します。
4. 初期候補と設定を確認し、状態を初期化します。

```sh
make cli ARGS='submission --config configs/submission/policy.yaml init'
make cli ARGS='submission --config configs/submission/policy.yaml init --execute'
make cli ARGS='submission --config configs/submission/policy.yaml plan'
```

5. GitHub Actionsの手動起動で未送信確認を行います。手動起動からは送信も状態更新も行いません。
6. リポジトリ変数`KAGGLE_SUBMISSION_ENABLED`を`true`にします。実送信は次の定時実行から有効です。

手元の`kaggle auth login`によるOAuth認証は、GitHub Actionsへ自動では引き継がれません。CLIが使えることと、定期実行用の認証が設定済みであることを区別してください。KaggleもCI/CD用途にはAPI Tokens設定で作成したトークンを案内しています。[公式の認証説明](https://www.kaggle.com/docs/api)

有効期限は最終結果確認の2026年12月4日20:15より後まで必要です。期限を確認できないトークンや、運用途中で失効する短期トークンのまま有効化しません。2026年10月22日の実接続では、OAuth認証からのトークン発行は最大16時間という応答でした。期限とSDKの変換結果の詳細は[設定確認の記録](../records/submission-setup-20261022.md)に残しています。

ワークフロー自体の状態が`active`でも、定期実行は`KAGGLE_SUBMISSION_ENABLED=true`のときだけ処理します。Secret・提出状態・GitHub上の未送信確認を揃えてから変数を設定してください。無効のままなら定時ジョブはスキップされます。

初期候補はworkflowの実行`20261021T014640Z-48c82b1c`です。CPU参照検証とGPUの開発用評価3/6を確認したZIPを、非公開Notebook `satter0927/gemma-4-developer-agent-baseline`の版10から取得します。開発用の3/6は正式スコアではありません。

状態は`codex/submission-state`ブランチに保存します。初期化・登録・試行記録は親コミットを指定し、fast-forward更新だけを許可します。競合や状態保存失敗時は送信しません。状態ブランチを削除・初期化し直すと重複防止の履歴が失われるため、復旧では既存履歴を保持します。

## 次の候補を登録する

現在のソースの事前検証ではなく、保存済みCPU・GPU実行の一致を確認します。候補ZIP、内容一覧、実行コード、設定、参照検証とその成果物、公式結果を照合します。

```sh
make cli ARGS='submission --config configs/submission/policy.yaml register --cpu-manifest CPU_MANIFEST --gpu-manifest GPU_MANIFEST --owner OWNER --kernel KERNEL --version VERSION --kernel-path OUTPUT_PATH/submission.zip --output configs/submission/candidates/NAME.json'
```

この操作は公開可能なハッシュと取得先を持つ候補記録を作成します。確認後、同じコマンドへ`--execute`を付けると状態ブランチへ登録します。登録順で処理し、同じ内容を再包装したZIPも重複候補として拒否します。

## 保留と受付不明

自動処理は手動提出から24時間保留し、UTC日付ごとの残枠も守ります。提出履歴から取得したZIPを照合し、手動で提出済みの候補も再送しません。受付済みで採点待ちの提出は、翌日の別候補の提出を妨げません。

正式受付の前に試行を保存します。タイムアウトや受付IDの欠落は受付不明として保持し、自動再送しません。次の同期で説明欄の試行識別子とZIPが一致すれば受付済みに更新します。

```sh
make cli ARGS='submission --config configs/submission/policy.yaml sync --execute'
make cli ARGS='submission --config configs/submission/policy.yaml resolve --token ATTEMPT_TOKEN --submission-id SUBMISSION_ID --execute'
```

受付されていないことを提出画面と履歴で確認した場合だけ、`--submission-id`の代わりに`--confirm-not-accepted`を指定します。推測で解除すると重複送信につながります。

## 結果と通知

実行要約と`artifacts/submission-status/report.json`へ、処理結果・受付ID・公開スコア上位2件を記録します。同点は新しい提出を優先します。最終選択のチェック状態は公式APIで確認できないため、未確認と表示します。締切前に提出画面で選択したIDを確認してください。上位の自動整理は最終選択の代行ではありません。

候補なし・手動提出による保留・受付完了・採点結果の更新は通知しません。既知の異常は状態へ記録し、初回だけ失敗通知の対象にします。継続中も要約には確認事項を残します。異常は処理ごとに保持し、該当する処理の回復を確認してから解除します。夜の同期成功だけで候補の異常を解除しません。未処理例外、状態保存失敗、ランナー障害は通常の失敗通知に従います。

GitHubの通知設定でActionsを「失敗したワークフローのみ通知」に設定します。アカウント全体に影響する設定のため自動変更しません。Codex側の定期監視は追加しません。[通知設定](https://docs.github.com/en/subscriptions-and-notifications/how-tos/managing-github-actions-notifications)

認証情報は提出ワークフローだけに渡します。ZIPとAPIの生レスポンスは一時領域で処理し、Actionsの成果物・キャッシュへアップロードしません。通常CIは合成入力だけを使い、Kaggleへの送信を行いません。

保存済みNotebookと提出履歴のZIPは、固定版SDKがリダイレクトを追従したHTTP応答から取得します。読み込み中にサイズ上限を確認し、受信が完了してから保存先へ反映します。サイズ超過・HTTPエラー・通信中断では不完全なZIPを削除し、既存ファイルを保持します。認証から取得完了までの時間は、転送用ワーカーの実行期限で制限します。

## 対比較で検証した候補と手動提出

対比較のCPU・GPUセッション、定義、候補、課題と順序、SDK、固定条件、対照、個別結果を照合します。各セッションが展開先とは別に保存したZIPと、指定した保存版NotebookのZIPが同一であることを確認します。ZIPを再生成したり、対比較から単独評価の記録を作って代用したりしません。

```sh
make cli ARGS='submission --config configs/submission/policy.yaml register-experiment --cpu-session CPU_SESSION --gpu-session GPU_SESSION --variant workflow-skills-v1 --owner OWNER --kernel KERNEL --version VERSION --kernel-path OUTPUT_PATH/archives/workflow-skills-v1/submission.zip --output configs/submission/candidates/NAME.json'
```

既定では候補記録の作成と取得・照合だけを行います。状態ブランチへの登録には同じコマンドへ`--execute`を追加します。全課題が完了し、比較と採用基準を満たすことを確認してから登録してください。

```sh
make cli ARGS='submission --config configs/submission/policy.yaml submit --candidate CANDIDATE_ID'
make cli ARGS='submission --config configs/submission/policy.yaml submit --candidate CANDIDATE_ID --execute'
```

最初のコマンドは未送信確認です。二つ目は指定候補だけを1回送信します。公式残枠、UTC日付の上限、重複、受付不明、保存競合、最終送信期限は定時処理と共通です。朝の時間制限と手動提出後24時間の保留だけは明示手動提出に適用しません。受付後の自動処理には手動提出として保留を反映します。受付不明なら再送せず、履歴との照合で解消します。
