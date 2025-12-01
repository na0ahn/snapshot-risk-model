import pandas as pd
import numpy as np


def process_oulad_data(
    assessments_path: str = "data/oulad/raw/assessments.csv",
    student_assessments_path: str = "data/oulad/raw/studentAssessment.csv",
    student_info_path: str = "data/oulad/raw/studentInfo.csv",
    output_filename: str = "data/oulad/processed/oulad_proxy_features.csv",
):
    """
    OULAD 원본 CSV들을 읽어서,
    (student × module × assessment) 단위의 tabular feature를 생성하는 함수.
    - 목표: 성대 Canvas 데이터와 맞춰 쓸 수 있는 F_core 스키마 템플릿 만들기.
    """

    # 1. 데이터 로드
    try:
        assessments = pd.read_csv(assessments_path)
        student_assessments = pd.read_csv(student_assessments_path)
        student_info = pd.read_csv(student_info_path)
    except FileNotFoundError as e:
        print(f"[오류] 데이터 파일을 찾을 수 없습니다: {e}")
        return

    print("[1] 데이터 로드 완료")
    print("  - assessments:", assessments.shape)
    print("  - studentAssessment:", student_assessments.shape)
    print("  - studentInfo:", student_info.shape)

    # 2. assessments 정리: date 없는 과제 제거 (Exam 등 일부)
    #    → deadline 기준 지각 계산이 불가능하므로 여기서는 제외
    assessments_clean = assessments.dropna(subset=["date"]).copy()

    # 3. merge: studentAssessment + assessments + studentInfo
    merged_df = pd.merge(
        student_assessments,
        assessments_clean,
        on="id_assessment",
        how="inner",
    )

    full_df = pd.merge(
        merged_df,
        student_info,
        on=["id_student", "code_module", "code_presentation"],
        how="inner",
    )

    print("[2] 병합 완료:", full_df.shape)

    # 4. 기본 레이블 및 시간 관련 값 계산
    # --------------------------------------------------
    # OULAD의 date, date_submitted는 "코스 시작일로부터의 day offset" 형태
    # -> 그대로 day offset으로 사용
    full_df["due_at_day_offset"] = full_df["date"]
    full_df["submitted_at_day_offset"] = full_df["date_submitted"]

    # days_diff > 0 이면 due 이후에 제출 → 지각
    full_df["days_diff"] = (
        full_df["submitted_at_day_offset"] - full_df["due_at_day_offset"]
    )
    full_df["is_late"] = (full_df["days_diff"] > 0).astype(int)

    # score NaN은 0점으로 처리 (미제출/Fail 등)
    if "score" in full_df.columns:
        full_df["score"] = full_df["score"].fillna(0.0)
    else:
        full_df["score"] = 0.0

    # weight 없는 경우 0으로
    if "weight" in full_df.columns:
        full_df["weight"] = full_df["weight"].fillna(0.0)
    else:
        full_df["weight"] = 0.0

    # 5. 과거 이력(History) feature 생성
    # --------------------------------------------------
    # 학생 × 모듈 × 프레젠테이션 단위로,
    # "현재 과제 이전까지"의 활동을 요약하는 feature를 만든다.
    group_keys = ["id_student", "code_module", "code_presentation"]

    full_df = full_df.sort_values(by=group_keys + ["due_at_day_offset"])
    grouped = full_df.groupby(group_keys, sort=False)

    # (1) 과거 평균 지연 시간
    full_df["history_avg_delay"] = grouped["days_diff"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # (2) 과거 지각 횟수
    full_df["history_late_count"] = grouped["is_late"].transform(
        lambda x: x.shift(1).expanding().sum()
    )

    # (3) 과거 점수 분산
    full_df["history_score_variance"] = grouped["score"].transform(
        lambda x: x.shift(1).expanding().std()
    )

    # (4) 과거 과제 수
    full_df["history_assessment_count"] = grouped["id_assessment"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    # (5) 과거 지각 비율
    full_df["history_late_ratio"] = np.where(
        full_df["history_assessment_count"] > 0,
        full_df["history_late_count"] / full_df["history_assessment_count"],
        0.0,
    )

    # (6) 과거 평균 점수
    full_df["history_avg_score"] = grouped["score"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # NaN 채우기
    history_cols = [
        "history_avg_delay",
        "history_late_count",
        "history_score_variance",
        "history_assessment_count",
        "history_late_ratio",
        "history_avg_score",
    ]
    full_df[history_cols] = full_df[history_cols].fillna(0.0)

    # 6. 코스 내 상대적 위치 (relative_due_position) 계산
    # --------------------------------------------------
    # 모듈-프레젠테이션별로 가장 마지막 마감일을 코스 길이로 보고,
    # 각 과제가 그 안에서 어느 위치(0~1)에 있는지 계산
    full_df["course_length_days"] = full_df.groupby(
        ["code_module", "code_presentation"]
    )["due_at_day_offset"].transform("max")

    full_df["relative_due_position"] = np.where(
        full_df["course_length_days"] > 0,
        full_df["due_at_day_offset"] / full_df["course_length_days"],
        0.0,
    )

    # 7. 성대 스키마와 맞추기 위한 rename + 컬럼 선택
    # --------------------------------------------------
    final_view = full_df.rename(
        columns={
            "id_assessment": "assignment_id",
            # TMA/CMA/Exam 등 과제 타입 → 성대의 task_type과 유사한 역할
            "assessment_type": "task_type",
            "weight": "assignment_weight",
            "score": "grade_score",
        }
    )

    columns_to_keep = [
        # 식별자
        "id_student",
        "code_module",
        "code_presentation",
        "assignment_id",

        # 과제/코스 메타
        "task_type",            # (TMA/CMA/Exam 등) → 성대의 quiz/exam/hw 등에 대응
        "assignment_weight",    # 성적 비중 → Canvas의 points_possible/weight 느낌
        "due_at_day_offset",
        "submitted_at_day_offset",
        "grade_score",
        "is_late",
        "relative_due_position",

        # 인구통계 (성대에서도 gender/age/disability와 매칭 가능)
        "gender",
        "age_band",
        "disability",

        # 과거 이력 feature
        "history_avg_delay",
        "history_late_count",
        "history_score_variance",
        "history_assessment_count",
        "history_late_ratio",
        "history_avg_score",
    ]

    # 실제로 존재하는 컬럼만 선택 (혹시 일부 컬럼이 없을 경우를 대비)
    columns_to_keep = [c for c in columns_to_keep if c in final_view.columns]

    result_df = final_view[columns_to_keep].copy()

    # 8. 저장
    result_df.to_csv(output_filename, index=False, encoding="utf-8")
    print(f"[완료] '{output_filename}' 파일이 생성되었습니다.")
    print("  - 최종 데이터 shape:", result_df.shape)
    print("  - 예시 5행:\n", result_df.head())


if __name__ == "__main__":
    process_oulad_data()
