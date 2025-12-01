import pandas as pd
import numpy as np

def build_snapshots_from_base(
    base_csv: str = "data/oulad/processed/oulad_proxy_features.csv",
    snapshot_offsets_days = (-7, -3, 0),
    output_filename: str = "data/oulad/processed/oulad_snapshots.csv",
):
    """
    per-assignment base 테이블(oulad_proxy_features.csv)을 받아서
    (student × assignment × snapshot) 단위의 row로 확장한다.

    snapshot_offsets_days:
        - 예: (-7, -3, 0)은 D-7, D-3, D-day 스냅샷을 의미
        - offset d에 대해:
            snapshot_abs_day = due_at_day_offset + d
            time_to_deadline_days = due_at_day_offset - snapshot_abs_day >= 0
        - time_to_deadline_days < 0 인 스냅샷은 drop한다 (due 이후는 의미 없으므로)
    """
    # 1. base CSV 로드
    try:
        base_df = pd.read_csv(base_csv)
    except FileNotFoundError as e:
        print(f"[오류] base CSV를 찾을 수 없습니다: {e}")
        return

    print("[1] base 데이터 로드 완료:", base_df.shape)
    required_cols = ["id_student", "assignment_id", "due_at_day_offset", "is_late"]
    for col in required_cols:
        if col not in base_df.columns:
            raise ValueError(f"[오류] base_df에 '{col}' 컬럼이 없습니다. 전처리 스텝을 확인하세요.")

    # 2. snapshot offsets 반복
    snapshot_dfs = []
    for offset in snapshot_offsets_days:
        tmp = base_df.copy()
        # snapshot 이 언제 찍힌 것인지 (due 기준 상대적인 offset, 예: -7)
        tmp["snapshot_day_rel"] = offset  # D-7 → -7, D-3 → -3, D-day → 0

        # 코스 시작일 기준 절대 day (OULAD 일자 오프셋 기준)
        tmp["snapshot_day_abs"] = tmp["due_at_day_offset"] + offset

        # 마감까지 남은 일수 (항상 >= 0인 스냅샷만 사용할 것)
        tmp["time_to_deadline_days"] = tmp["due_at_day_offset"] - tmp["snapshot_day_abs"]

        # due 이후 스냅샷(음수 남은 일수)은 드랍
        before_due_mask = tmp["time_to_deadline_days"] >= 0
        num_dropped = (~before_due_mask).sum()
        if num_dropped > 0:
            print(f"  - offset {offset}: due 이후 스냅샷 {num_dropped}개 제거")

        tmp = tmp[before_due_mask].copy()
        snapshot_dfs.append(tmp)

    # 3. concat 해서 전체 snapshot 테이블 생성
    if not snapshot_dfs:
        print("[경고] 생성된 snapshot 데이터가 없습니다. snapshot_offsets_days 설정을 확인하세요.")
        return

    snapshots_df = pd.concat(snapshot_dfs, axis=0, ignore_index=True)

    # 4. snapshot 고유 ID / label 컬럼 추가
    #    - label_late: 모델용 타겟 컬럼 이름 (is_late 복사)
    snapshots_df["label_late"] = snapshots_df["is_late"].astype(int)

    # (선택) 각 row에 고유 snapshot_id 부여
    snapshots_df["snapshot_id"] = np.arange(len(snapshots_df))

    # 정렬 (학생, 과제, snapshot 순)
    snapshots_df = snapshots_df.sort_values(
        by=["id_student", "assignment_id", "time_to_deadline_days"],
        ascending=[True, True, False]  # 마감까지 남은 일수: 많→적 순으로 보고 싶으면 False
    ).reset_index(drop=True)

    # 5. 저장
    snapshots_df.to_csv(output_filename, index=False, encoding="utf-8")
    # print(f"[완료] '{output_filename}' 파일이 생성되었습니다.")
    # print("  - 최종 snapshot 데이터 shape:", snapshots_df.shape)
    # print("  - 예시 5행:\n", snapshots_df.head())


if __name__ == "__main__":
    # 1단계: 이미 oulad_proxy_features.csv 가 있다면 이 줄은 생략해도 OK
    # from oulad_preprocess import process_oulad_data
    # process_oulad_data()

    # 2단계: snapshot 생성
    build_snapshots_from_base(
        base_csv="data/oulad/processed/oulad_proxy_features.csv",
        snapshot_offsets_days=(-7, -3, 0),  # 필요하면 (-7, -3, -1, 0) 등으로 조정 가능
        output_filename="data/oulad/processed/oulad_snapshots.csv",
    )

