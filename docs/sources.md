# 公式資料と取得元

2026年9月30日に、以下の公式資料と配布ファイルを確認しました。

- [コンペ概要](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)
- [競技ルール](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules)
- [競技データ](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)
- [公式スターター](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)
- [公式Wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)
- [Streamlitの画面テスト](https://docs.streamlit.io/develop/api-reference/app-testing)

`agent/`の初期設定とプロンプトは競技データの`sample_submission`に由来します。配布LoRAは取得元に残し、Kaggle上で読み込みます。サンプリング設定は公式スターターと同じ内容です。第三者の資料・モデル・コードの権利を、このリポジトリのコードと一括して再許諾しません。

確認した公式パッケージは、swegemma 0.2.7、adk-submission 0.2.11、adk-eval-core 0.1.0です。実行時には版の一致を確認し、使用するWheelhouseのファイル名とSHA-256を記録します。

コンペ終了後に公開する際は、独自コード、公式サンプル、モデル、競技データ、実行ログごとに公開可能な範囲を確認します。
