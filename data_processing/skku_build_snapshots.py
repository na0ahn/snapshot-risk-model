# data_processing/skku_build_snapshots.py
import os
import sys

# 상위 폴더(project_root)를 path에 추가하여 모듈 import가 가능하도록 함 (직접 실행 시)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.oulad_build_snapshots import build_snapshots_from_base

def build_skku_snapshots(
    base_csv: str = "data/skku/processed/skku_proxy_features.csv",
    output_csv: str = "data/skku/processed/skku_snapshots.csv",
    snapshot_offsets_days = (-7, -3, 0),
):
    """
    SKKU Base Table -> Snapshot Table 변환
    (OULAD와 동일한 로직 사용)
    """
    if not os.path.exists(base_csv):
        print(f"[오류] Base 파일이 없습니다: {base_csv}")
        print("먼저 skku_process.py를 실행하세요.")
        return

    print(f"SKKU Snapshot 생성 시작: {snapshot_offsets_days}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    build_snapshots_from_base(
        base_csv=base_csv,
        snapshot_offsets_days=snapshot_offsets_days,
        output_filename=output_csv,
    )

if __name__ == "__main__":
    build_skku_snapshots()