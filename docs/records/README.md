# 実験・検証記録

| 日付 | 記録 | 内容 |
|---|---|---|
| 2026-10-20〜21 | [評価・診断](validation-20261020-21.md) | 初回候補、診断、固定6問と12結果、当時のコード・設定・取得先 |
| 2026-10-21 | [公開情報の調査](research-20261021.md) | 取得した公開手法、正式スコア、SDK差、当時の証拠 |
| 2026-10-21 | [Easy課題の実験準備](easy-experiments-20261021.md) | 固定した課題群、逐次構成、CPU検証とMLflow同期 |
| 2026-10-21 | [実験管理と改善機能の検証](agent-improvements-20261021.md) | 公開記録、新規6問の対照、Skill、GEPA、ログ |
| 2026-10-22 | [評価基盤とSkill比較](skill-comparison-20261022.md) | 監視・ログ・分析の修正、Kaggle CPU対照、部分比較と採用保留 |
| 2026-10-22 | [定期提出の設定確認](submission-setup-20261022.md) | 未有効化の確認、既存候補の照合、OAuthの期限制約、未設定への復帰 |
| 2026-10-21 | [配置変更の検証](structure-20261021.md) | 新配置の検証と移行手順 |

過去の記録にあるパスとコマンドは当時の配置です。現在の操作は運用文書に従います。記録の値や過去の生成物は変更せず、新しい検証を別に残します。

旧`configs/evaluation/5min.yaml`は過去の5分・10ツール条件、旧`5min-30tools.yaml`は固定6問の5分・30ツール比較条件でした。前者は現行プロファイルから参照されず削除し、後者は`configs/evaluation/comparison.yaml`へ移しました。正式提出物の`agent/eval_config.yaml`は維持します。以前の生成予算・LoRA比較の結果は評価・診断記録へ保持しています。

CPUで確認した条件と観測は、[検証Skill](experiments/easy-skills-v3-cpu-20261022.json)と[GEPAの合成探索](experiments/easy-gepa-v2-cpu-20261022.json)にも記載しています。CPU検証はモデルによる解決率を測定していません。[SkillのGPU比較](experiments/easy-skills-v3-gpu-20261022.json)は予定6問のうち3問で両候補とも2問成功し、残り3問は未実行のため採用保留です。
