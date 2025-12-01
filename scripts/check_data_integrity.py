# scripts/check_data_integrity.py
import os
import pandas as pd
from configs.feature_config import NUMERIC_COLS, CAT_COLS, LABEL_COL, GROUP_COLS

def check_base_csv(path: str, name: str):
    print(f"\n[Base Check] {name}: {path}")
    if not os.path.exists(path):
        print("  ✗ 파일이 없습니다.")
        return

    df = pd.read_csv(path)
    print(f"  ✓ 로드 성공. shape = {df.shape}")

    required_cols = [
        "id_student", "code_module", "code_presentation", "assignment_id",
        "task_type", "assignment_weight",
        "due_at_day_offset", "submitted_at_day_offset", "relative_due_position",
        "grade_score", "is_late",
        "gender", "age_band", "disability",
        "history_avg_delay", "history_late_count", "history_assessment_count",
        "history_late_ratio", "history_avg_score", "history_score_variance",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("  ✗ 누락된 컬럼:", missing)
    else:
        print("  ✓ 모든 필수 컬럼 존재")

    # NaN 비율 상위 10개
    na_ratio = df.isna().mean().sort_values(ascending=False).head(10)
    print("  ▷ NaN 비율 Top 10:")
    print(na_ratio)

    # 레이블 분포 (있으면)
    if "is_late" in df.columns:
        print("  ▷ is_late 분포:")
        print(df["is_late"].value_counts(dropna=False))


def check_snapshot_csv(path: str, name: str, expected_offsets=(-7, -3, 0)):
    print(f"\n[Snapshot Check] {name}: {path}")
    if not os.path.exists(path):
        print("  ✗ 파일이 없습니다.")
        return

    df = pd.read_csv(path)
    print(f"  ✓ 로드 성공. shape = {df.shape}")

    required_cols = NUMERIC_COLS + CAT_COLS + [LABEL_COL, "snapshot_day_rel", "snapshot_day_abs", "time_to_deadline_days"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("  ✗ 누락된 컬럼:", missing)
    else:
        print("  ✓ 모델 입력/타겟에 필요한 컬럼 모두 존재")

    # snapshot offset 확인
    if "snapshot_day_rel" in df.columns:
        print("  ▷ snapshot_day_rel unique:", sorted(df["snapshot_day_rel"].unique()))
    if "time_to_deadline_days" in df.columns:
        print("  ▷ time_to_deadline_days 범위:",
              df["time_to_deadline_days"].min(), "→", df["time_to_deadline_days"].max())

    # 그룹당 snapshot 개수 대략 확인
    if all(c in df.columns for c in GROUP_COLS):
        group_counts = df.groupby(GROUP_COLS).size()
        print("  ▷ 그룹당 snapshot 개수 통계:")
        print(group_counts.describe())


if __name__ == "__main__":
    # 경로는 네가 정한 구조에 맞춰 수정
    check_base_csv("data/oulad/processed/oulad_proxy_features.csv", "OULAD Base")
    check_snapshot_csv("data/oulad/processed/oulad_snapshots.csv", "OULAD Snapshots")

    check_base_csv("data/skku/processed/skku_proxy_features.csv", "SKKU Base")
    check_snapshot_csv("data/skku/processed/skku_snapshots.csv", "SKKU Snapshots")
