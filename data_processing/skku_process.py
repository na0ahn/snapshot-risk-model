# data_processing/skku_process.py
import os
import glob
import json
import pandas as pd
import numpy as np

def map_demographics(user_info: dict):
    """SKKU user_info를 OULAD 포맷(gender, disability, age_band)으로 변환."""
    gender_map = {"남": "M", "여": "F"}
    gender = gender_map.get(user_info.get("gender"), "M")

    disability = "N" if user_info.get("disabled") == "아니요" else "Y"

    age = user_info.get("age", 20)
    if age < 35: age_band = "0-35"
    elif age < 55: age_band = "35-55"
    else: age_band = "55<="

    return gender, disability, age_band

def classify_task_type(name: str) -> str:
    """과제 이름 기반 Task Type 분류."""
    name = str(name).lower()
    if any(x in name for x in ["quiz", "퀴즈", "쪽지시험"]): return "Quiz"
    if any(x in name for x in ["exam", "midterm", "final", "고사", "시험"]): return "Exam"
    if any(x in name for x in ["project", "team", "팀", "프로젝트"]): return "Project"
    return "Homework"

def process_skku_json_files(
    raw_dir: str = "data/skku/raw",
    output_path: str = "data/skku/processed/skku_proxy_features.csv",
):
    # 1. 파일 탐색
    pattern = os.path.join(raw_dir, "*_canvas_all_assignments_with_submissions.json")
    file_paths = sorted(glob.glob(pattern))

    if not file_paths:
        print(f"[경고] '{pattern}' 경로에 파일이 없습니다. 경로를 확인하세요.")
        return None

    print(f"총 {len(file_paths)}개의 JSON 파일 처리 시작...")
    all_assignments = []

    # 2. JSON 로드 및 병합
    for idx, fpath in enumerate(file_paths):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        user_info = data.get("user_info", {})
        assignments = data.get("assignments", [])
        exported_at = data.get("exported_at")

        gender, disability, age_band = map_demographics(user_info)

        for asm in assignments:
            row = asm.copy()
            # 파일별 가상 ID 부여 (추후 실제 학번 매핑 가능)
            row["id_student"] = 10000 + idx
            row["gender"] = gender
            row["disability"] = disability
            row["age_band"] = age_band
            row["exported_at"] = exported_at
            all_assignments.append(row)

    df = pd.DataFrame(all_assignments)
    
    # 3. 날짜 변환
    date_cols = ["due_at", "submitted_at", "unlock_at", "course_start_at", "exported_at"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 4. Task Type & Weight
    df["task_type"] = df["assignment_name"].apply(classify_task_type)
    df["assignment_weight"] = pd.to_numeric(df["points_possible"], errors='coerce').fillna(0.0)

    # 5. Course & Presentation
    df["code_module"] = df["course_id"].astype(str)
    if "term_name" not in df.columns: df["term_name"] = "Unknown"
    df["code_presentation"] = df["term_name"].fillna("Unknown")

    # 6. Time Offset Calculation
    # 코스 시작일 추정: course_start_at -> unlock_at min -> due_at min
    df["temp_start"] = df["course_start_at"]
    
    if "unlock_at" in df.columns:
        course_min_unlock = df.groupby(["code_module", "code_presentation"])["unlock_at"].transform("min")
        df["temp_start"] = df["temp_start"].fillna(course_min_unlock)
        
    course_min_due = df.groupby(["code_module", "code_presentation"])["due_at"].transform("min")
    df["temp_start"] = df["temp_start"].fillna(course_min_due)

    # Offset (Days)
    df["due_at_day_offset"] = (df["due_at"] - df["temp_start"]).dt.days
    df["submitted_at_day_offset"] = (df["submitted_at"] - df["temp_start"]).dt.days

    # Relative Position (0~1)
    course_length = df.groupby(["code_module", "code_presentation"])["due_at_day_offset"].transform("max")
    df["relative_due_position"] = (df["due_at_day_offset"] / course_length).fillna(0.0)

    # 7. Labeling: is_late (제출 지각 + 미제출 지연 포함)
    df["is_late"] = 0
    mask_due_defined = df["due_at"].notna()
    
    if "has_submitted_submissions" not in df.columns:
        df["has_submitted_submissions"] = False # 컬럼 없으면 False 처리
        
    mask_submitted = df["has_submitted_submissions"] == True

    # (1) 제출함 & 지각함
    if "submission_late" in df.columns:
        df.loc[mask_submitted & mask_due_defined, "is_late"] = (
            df.loc[mask_submitted & mask_due_defined, "submission_late"]
            .fillna(False).astype(int)
        )
    else:
        # submission_late 컬럼 없으면 날짜 비교
        submitted_late_mask = df["submitted_at"] > df["due_at"]
        df.loc[mask_submitted & mask_due_defined & submitted_late_mask, "is_late"] = 1

    # (2) 미제출 & 이미 마감 지남 (현재 시점 exported_at 기준)
    mask_unsubmitted = (~mask_submitted) & mask_due_defined
    mask_past_due = mask_unsubmitted & (df["exported_at"] > df["due_at"])
    df.loc[mask_past_due, "is_late"] = 1

    # 8. Days Diff (제출 지연일 + 미제출 경과일)
    # 기본: 제출일 - 마감일
    df["days_diff"] = (df["submitted_at"] - df["due_at"]).dt.total_seconds() / (24 * 3600)

    # 미제출인 경우: 현재(exported_at) - 마감일
    # (아직 마감 안 지났으면 음수, 지났으면 양수)
    mask_missing_submit = df["submitted_at"].isna() & mask_due_defined & df["exported_at"].notna()
    df.loc[mask_missing_submit, "days_diff"] = (
        (df.loc[mask_missing_submit, "exported_at"] - df.loc[mask_missing_submit, "due_at"])
        .dt.total_seconds() / (24 * 3600)
    )
    df["days_diff"] = df["days_diff"].fillna(0.0)

    # 9. Grade Score (우선순위: submission_score -> score -> 0)
    if "submission_score" in df.columns:
        df["grade_score"] = df["submission_score"].fillna(0.0)
    elif "score" in df.columns:
        df["grade_score"] = df["score"].fillna(0.0)
    else:
        df["grade_score"] = 0.0

    # 10. History Feature Engineering
    print("[2] History Feature 생성 중...")
    df = df.sort_values(by=["id_student", "code_module", "code_presentation", "due_at"])
    grouped = df.groupby(["id_student", "code_module", "code_presentation"])

    # Shift(1) & Expanding
    df["history_avg_delay"] = grouped["days_diff"].transform(lambda x: x.shift(1).expanding().mean())
    df["history_late_count"] = grouped["is_late"].transform(lambda x: x.shift(1).expanding().sum())
    df["history_assessment_count"] = grouped["assignment_id"].transform(lambda x: x.shift(1).expanding().count())
    df["history_avg_score"] = grouped["grade_score"].transform(lambda x: x.shift(1).expanding().mean())
    df["history_score_variance"] = grouped["grade_score"].transform(lambda x: x.shift(1).expanding().std())

    df["history_late_ratio"] = np.where(
        df["history_assessment_count"] > 0,
        df["history_late_count"] / df["history_assessment_count"],
        0.0,
    )

    hist_cols = ["history_avg_delay", "history_late_count", "history_assessment_count", 
                 "history_late_ratio", "history_avg_score", "history_score_variance"]
    df[hist_cols] = df[hist_cols].fillna(0.0)

    # 11. Select Columns & Save
    columns_to_keep = [
        "id_student", "code_module", "code_presentation", "assignment_id",
        "task_type", "assignment_weight", 
        "due_at_day_offset", "submitted_at_day_offset", "relative_due_position",
        "grade_score", "is_late", 
        "gender", "age_band", "disability",
        "history_avg_delay", "history_late_count", "history_assessment_count",
        "history_late_ratio", "history_avg_score", "history_score_variance"
    ]

    final_cols = [c for c in columns_to_keep if c in df.columns]
    result_df = df[final_cols].copy()
    
    # 마감일이 없는 과제(이벤트성 등)는 제외
    result_df = result_df[result_df["due_at_day_offset"].notna()].reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[완료] '{output_path}' 생성됨. Shape: {result_df.shape}")

    return result_df

if __name__ == "__main__":
    process_skku_json_files()