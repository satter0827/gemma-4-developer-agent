# パッチ診断

保存された修正の失敗原因を、モデルを起動せずに調べます。ローカルのPythonで繰り返し確認し、最後に同じパッチをKaggle CPUの公式検証へ渡します。エージェントが修正を生成する性能の評価とは別の作業です。

## 準備と実行

```sh
make setup
make cli ARGS='prepare-diagnostics --config configs/diagnostics/fastapi.yaml'
make cli ARGS='diagnose --config configs/diagnostics/fastapi.yaml'
```

準備には、認証済みKaggle CLI、コンペデータへのアクセス、uv、Python 3.12を使用します。公式スナップショット2件は合計約636MBで、初回だけ取得します。スナップショット・課題データ・生成パッチを設定内のSHA-256と照合します。配布スナップショットのGit履歴は再構成されているため、課題データの元コミットとスナップショットのHEADをそれぞれ記録します。

課題データと診断対象の保存パッチは、設定で指定した場所にあらかじめ配置します。これらの競技データや過去の実行結果はGit管理対象に含めません。欠損やハッシュ不一致がある場合は準備を停止します。

依存関係は`configs/environments/diagnostics.in`を元に、ハッシュ付きの`configs/environments/diagnostics.lock`で固定します。更新するときだけロックファイルを作り直し、準備と診断を再実行します。課題ごとの仮想環境は`artifacts/task-envs/`に置き、開発用環境へ課題の依存関係を追加しません。Pythonの実際の版、CPUの種類、導入済みパッケージも記録し、各ケースの終了後に環境の一致を確認します。

通常の診断は入力・ソース・依存関係の一致を確認してから実行します。ダウンロードや依存更新は行いません。原本は`artifacts/reference/diagnostics/`に保持し、一時作業コピーは各ケースの終了時に削除します。

## 設定

共通の課題定義は`configs/tasks/catalog.yaml`に置き、課題入力・スナップショット・テスト対象・ソースルート・import対象・追加依存のハッシュを管理します。診断設定は課題IDを参照し、生成パッチと原因確認用の置換を別に持ちます。Kaggle評価の課題順は固定済みの選定記録から読みます。

診断設定には、共通課題定義、生成パッチ、CPU依存の入力と固定ファイル、タイムアウト、保存先、Private NotebookのIDを指定します。診断用パッチは生成パッチへの回数を限定した文字列置換として定義します。置換元が変わった場合は実行を停止します。課題IDごとの分岐はコードに持ちません。

準備開始前に入力ファイルの存在と保存先の分離を検証します。キャッシュ、課題用環境、診断結果は互いに重ねず、入力ファイル、開発環境、公式評価結果の保存先とも分けます。診断設定のタイムアウトは、ローカルとKaggle CPUの両方へ渡します。

## 4つの条件

各課題で、参照パッチ、未修正、生成パッチ、原因を一つ変えた診断パッチを順番に検証します。参照パッチが通らなければ残りの検証を停止し、環境の問題と生成パッチの品質を混同しません。

SSE課題では、例外文言だけを変更した結果を確認します。リリース課題では、ファイル名だけを変更して公開APIの不足を確認します。採点用テストを変更したり、参照パッチの内容をエージェントへ与えたりしません。

## 新しい生成パッチの再診断

診断設定を別名で作り、`generated_patch`と`generated_sha256`を新しいパッチへ変更します。`generated_expected_pass`にはGPUの公式判定を指定し、収集失敗が期待されない場合は`collection_error_cases`を空リストにします。`probe_replacements`を空リストにすると診断用の差分を加えずに再現できます。キャッシュ・課題用環境・保存先も別の診断用領域を指定し、元の事前検証記録を変更しません。環境入力が同じなら準備を再利用し、診断を実行します。参照パッチの通過と新しいパッチの結果を記録し、公式の成績へ加算しません。

## 結果

`artifacts/diagnostics/実行ID/`に`report.json`と、ケースごとの`case.json`、`test.log`、`junit.xml`、適用パッチ、`source-guard/origins.json`を保存します。処理開始後の失敗も記録します。テストが実行されなかった場合はJUnitが未生成の場合があります。

入力パッチは元の内容とハッシュを保存し、末尾改行の補完は適用時だけ行います。ログ・JUnit・パッチにもハッシュを付け、CPU用コードの生成前に欠損・変更・ケースの重複を検出します。タイムアウト時は途中のログと終了コードを残し、子プロセスも終了してから作業コピーを削除します。

`passed`はそのケースのテスト通過、`matched`は期待した結果との一致です。未修正や生成パッチの失敗を再現できたケースでは、`passed=false`、`matched=true`となります。全ケースの期待結果が一致したときだけ診断を完了とします。通常のテスト結果は終了コード0・1を対象とし、終了コード2、テスト0件、全件スキップ、タイムアウト、適用失敗、保存失敗をテスト実行の成功として扱いません。

モジュール不足など、収集失敗自体を再現するケースは`collection_error_cases`へ明示します。この場合もテスト件数は0件で、JUnitの合成ケースを実行テストへ数えません。`outcome=collection_error`と`collection_errors`を別に記録し、通常の失敗と区別します。参照パッチにこの指定はできず、実テストの通過が必須です。中断や通常テストが混在したJUnitは、収集失敗の証拠として受け入れません。

診断の完了は終了コード0、失敗は1、引数エラーは2です。公式の解決数や解決率は変更しません。

## Kaggle CPUで確認する

```sh
make cli ARGS='prepare-kaggle --name diagnostic-cpu-20261020 --diagnostics-config configs/diagnostics/fastapi.yaml'
kaggle kernels push -p artifacts/kaggle/diagnostic-cpu-20261020
kaggle kernels status satter0927/gemma-4-developer-agent-diagnostics
kaggle kernels logs satter0927/gemma-4-developer-agent-diagnostics --follow
kaggle kernels output satter0927/gemma-4-developer-agent-diagnostics -p artifacts/diagnostics/kaggle-cpu-20261020 --page-size 200 --file-pattern '^artifacts/diagnostics/.*'
```

ローカルで8条件の診断が完了し、入力が変わっていない場合だけCPU用コードを生成できます。生成先は毎回新しい名前を指定します。NotebookはPrivate、GPU・TPU・インターネット接続は無効で、モデル重みを入力へ追加しません。ソースと全パッチのハッシュを保存します。

Kaggleでは公式の`verify_task`をそのまま使用します。公式処理が利用するJUnitも、サンドボックスの終了前に保存し、終了コードとの一致を確認します。JUnitの欠損、テスト0件、全件スキップ、途中停止は診断の失敗として記録します。MacとKaggleではOS・依存関係・テスト環境の準備が異なるため、記録した環境とケース結果を照合します。自動再試行やGPU評価への自動移行はありません。

## 読み込み元と比較課題

テストと同じプロセスで、対象パッケージとサブモジュールの読み込み元を記録します。`src`配置では設定されたソースルートを使い、インストール済みの別の版を読んだ結果を成功にしません。外部パッケージへの正規の別名は、実際のモジュール名で区別します。

比較用の診断設定では生成パッチとそのハッシュを明示的にnull、置換を空にし、未修正・参照パッチの2条件だけを実行できます。課題の選定には`qualify --config configs/tasks/comparison.yaml`を使い、事前に決めた順位、除外理由、各条件の証拠を保存します。モデルの結果から課題を選び直しません。

公式SDKの開発用subprocessでは、ソースルートとハッシュで固定した不足依存だけを補います。追加依存は課題ごとの仮想環境にオフラインで導入し、公式の採点関数やテストを変更しません。MacのCPU依存固定とKaggleの公式Wheel・追加Wheel固定は別に管理し、各環境の実際の依存一覧も残します。

固定済み比較6問の再検証は`qualify --config configs/tasks/comparison.yaml --fixed-selection`を使います。新しい記録と`current.json`を作り、旧選定記録を保持します。失敗した課題を補欠へ入れ替えません。
