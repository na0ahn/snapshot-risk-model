# configs/feature_config.py

# 1. 모델 입력으로 사용할 Numeric Feature (F_core)
# - 주의: grade_score, submitted_at 등 미래 정보/정답에 가까운 값은 절대 포함하지 않음.
NUMERIC_COLS = [
    # 과제 메타
    "assignment_weight",       # 과제 비중
    "due_at_day_offset",       # 마감일 (코스 시작 기준)
    "relative_due_position",   # 학기 중 과제 위치 (0~1)

    # 과거 이력 요약 (Current Assignment 제외, 이전까지의 누적)
    "history_avg_delay",       # 평균 지연 일수
    "history_late_count",      # 지각 횟수
    "history_score_variance",  # 점수 변동성
    "history_assessment_count",# 과제 수행 횟수
    "history_late_ratio",      # 지각 비율
    "history_avg_score",       # 평균 점수

    # Snapshot 시점 정보
    "snapshot_day_rel",        # D-day 기준 상대 일수 (-7, -3, 0)
    "snapshot_day_abs",        # 절대 스냅샷 일수
    "time_to_deadline_days",   # 마감까지 남은 일수 (중요: Monotone Loss의 기준)
]

# 2. 모델 입력으로 사용할 Categorical Feature (F_core)
CAT_COLS = [
    "code_module",         # 과목 코드
    "code_presentation",   # 학기 구분
    "task_type",           # 과제 유형 (Quiz, Exam, Homework ...)
    "gender",              # 성별
    "age_band",            # 연령대
    "disability",          # 장애 여부
]

# 3. 타겟 및 그룹핑 정보
LABEL_COL = "label_late"         # 타겟: 지각 여부 (0/1)
GROUP_COLS = ["id_student", "assignment_id"] # Monotone Loss를 위한 그룹 (학생-과제)

# 4. (선택) 성균관대 전용 추가 피처 (나중에 사용)
SKKU_NUMERIC_EXTRA = [
    # "midterm_score", 
    # "slump_hw",
    # "buffer_to_due" 
] 