"""Read-only local UI for official evaluation artifacts."""

import os
from pathlib import Path

import streamlit as st

from gemma_agent_dev.results import Run, comparison_differences, load_run, read_artifact

CONDITION_LABELS = {
    "task_ids": "対象課題",
    "evaluation": "実行上限",
    "environment": "実行環境",
    "tasks_sha256": "課題データ",
    "model_source": "モデル",
    "docker_image": "実行イメージ",
    "vllm": "推論設定",
    "context_cache": "キャッシュ設定",
    "compaction": "履歴の圧縮設定",
    "incomplete": "完了状況",
}


def condition(run: Run, field: str) -> object:
    if field == "incomplete":
        return "完了" if run.complete else "未完了・要確認"
    return run.manifest.get(field, run.manifest.get("runtime_config", {}).get(field))


def status(run: Run, task_id: str) -> str:
    row = run.rows.get(task_id)
    if row is None:
        return "未完了"
    if row["resolved"]:
        return "解決"
    return "実行エラー" if row.get("error") else "未解決"


def show_artifact(run: Run, relative: str) -> None:
    try:
        st.code(read_artifact(run, relative), language="text")
    except (OSError, ValueError):
        st.info("このファイルは保存されていません。")


def main() -> None:
    st.set_page_config(page_title="Gemma 評価結果", layout="wide")
    st.title("Gemma 評価結果")
    directory = Path(os.environ.get("GEMMA_RESULTS_DIR", "results"))
    st.caption(f"結果フォルダー：{directory}")
    runs: list[Run] = []
    for path in sorted(directory.glob("**/manifest.json"), reverse=True):
        try:
            runs.append(load_run(path.parent))
        except (OSError, ValueError, TypeError) as error:
            st.error(f"結果を読み込めません：{path.parent.name}（{error}）")
    if not runs:
        st.info("評価結果はまだありません。Kaggleの実行結果を取得すると表示されます。")
        return
    st.dataframe(
        [
            {
                "実験": run.path.name,
                "状態": "完了" if run.complete else "未完了・要確認",
                "予定": len(run.task_ids),
                "完了": len(run.rows),
                "解決": run.resolved,
                "解決率": f"{run.resolution_rate:.1%}" if run.complete else "—",
                "所要時間（秒）": run.manifest.get("timings", {}).get("total_seconds"),
            }
            for run in runs
        ],
        hide_index=True,
        width="stretch",
    )
    index = st.selectbox("実験を選択", range(len(runs)), format_func=lambda i: runs[i].path.name)
    run = runs[index]
    if not run.complete:
        st.warning("完了していない結果、または確認が必要な結果です。解決率の比較対象にしません。")
    if run.manifest.get("error"):
        st.error(run.manifest["error"])
    if run.problems:
        with st.expander("結果ファイルの確認事項", expanded=True):
            for problem in run.problems:
                st.text(problem)
    with st.expander("実行条件"):
        st.json(run.manifest)
    task_id = st.selectbox("課題を選択", run.task_ids)
    st.write(f"結果：{status(run, task_id)}")
    if task_id in run.rows:
        st.json(run.rows[task_id])
    safe_id = task_id.replace("/", "__")
    for title, relative in (
        ("変更パッチ", f"patches/{safe_id}.patch"),
        ("検証ログ", f"test_outputs/{safe_id}.log"),
        ("実行ログ", f"logs/{safe_id}.log"),
        ("実行トレース", f"traces/trace_{safe_id}.json"),
    ):
        with st.expander(title):
            show_artifact(run, relative)
    if len(runs) > 1:
        st.subheader("実験の比較")
        other_index = st.selectbox(
            "比較対象",
            [i for i in range(len(runs)) if i != index],
            format_func=lambda i: runs[i].path.name,
        )
        other = runs[other_index]
        differences = comparison_differences(run, other)
        if differences:
            st.warning("比較条件が異なるか、結果が未完了です。単純な優劣は判定できません。")
            st.write("異なる条件：" + "、".join(CONDITION_LABELS[key] for key in differences))
            with st.expander("条件の差を確認"):
                for key in differences:
                    st.write(CONDITION_LABELS[key])
                    st.json(
                        {"選択した実験": condition(run, key), "比較対象": condition(other, key)}
                    )
        else:
            st.success("同じ課題と実行条件で比較できます。")
        st.dataframe(
            [
                {
                    "課題": task,
                    "選択した実験": status(run, task),
                    "比較対象": status(other, task),
                    "選択した実験（秒）": run.rows.get(task, {}).get("duration_seconds"),
                    "比較対象（秒）": other.rows.get(task, {}).get("duration_seconds"),
                }
                for task in sorted(set(run.task_ids) | set(other.task_ids))
            ],
            hide_index=True,
            width="stretch",
        )


main()
