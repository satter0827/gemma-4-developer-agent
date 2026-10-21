# 環境の準備

Python 3.12とuvを使用します。開発環境は`artifacts/venv`へ置き、課題用環境・公式SDK環境と分離します。

```sh
make setup
make check
make setup-submission
make verify
```

最後の2操作は提出SDKの契約も確認する場合です。開発用ソースを変更した後は`make check`でインストール内容を更新します。

## CPU環境

競技データ、保存パッチ、公式Wheel、スナップショットは認証済みの取得済み入力を使います。Gitへ追加しません。

```sh
make cli ARGS='prepare-diagnostics --config configs/diagnostics/fastapi.yaml'
make cli ARGS='prepare-diagnostics --config configs/diagnostics/comparison.yaml'
make cli ARGS='prepare-runtime-checks --config configs/environments/sdk-cpu.yaml'
```

準備処理だけが取得・依存導入を行います。取得済み入力とuvキャッシュが揃う環境では`UV_OFFLINE=1`を指定できます。Python、依存、Wheel、スナップショットが変われば再準備します。推論条件だけが変わった場合は環境を作り直さず、新しい実行記録を作ります。

## 公式SDKと台帳

```sh
make cli ARGS='check-runtime --config configs/environments/sdk-cpu.yaml'
make cli ARGS='task-report --config configs/tasks/report.yaml --write-docs'
make cli ARGS='task-report --config configs/tasks/report.yaml --observations-config configs/evaluation/observations.yaml --write-docs'
```

SDK確認は合成入力で生成リクエスト・登録・コンパイルを検査します。重みやGPUを使いません。台帳の履歴結合は任意です。指定した履歴の欠損・改変は無視せず失敗します。`--write-docs`の更新先は`docs/reference/tasks.md`だけです。

`make`だけでコマンド一覧を表示します。`make format`は整形を行い、`check`と`verify`は整形の検査だけを行います。`check`は環境更新の終了後に`verify`を開始し、並列指定時も同じ一時保存先でテストを重ねません。
