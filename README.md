-----

# SKKU Assignment Risk Prediction (Teacher-Student Framework)

## 프로젝트 개요

이 프로젝트는 성균관대학교 학생들의 과제 제출 지연(Risk)을 예측하기 위해 개발되었습니다.
데이터 부족(Small Data) 문제를 해결하기 위해 **OULAD 대용량 데이터로 학습된 Teacher 모델의 지식을 전수받는 Knowledge Distillation(KD)** 기법을 적용했습니다.

-----

## Quick Start (실행 방법)


```bash
python inference_demo.py
```
해당 코드를 참고하여 세가지 모델을 비교하거나, 한가지에 초점을 맞추어 script를 짜 실험해보면 좋을 것 같습니다.

  * **기능:** `best_mlp_student.pt` 가중치를 로드하여 샘플 학생들의 위험도(Risk Probability)와 상태(Danger/Warning/Safe)를 출력합니다.
  * **설정:** `configs/best_params.py`의 `DEFAULT_CONFIG`를 따릅니다.

-----

## 핵심 실험 가이드 (Experiments)

### 1\. 딥러닝 모델 전략 비교 (Strategy Comparison)

데이터 규모에 맞는 최적의 딥러닝 구조를 찾기 위한 실험입니다.
  * **비교 모델:**
    1.  **Teacher-Like (Transformer):** Teacher와 유사한 구조. 표현력은 좋으나 데이터가 적어 과적합 위험.
    2.  **Lightweight (Efficient):** 파라미터 없는 Identity Attention 사용. 극소량 데이터에 강건함.
    3.  **MLP Student (Best):** Teacher의 임베딩(지식)만 가져오고, 구조를 단순화(MLP)한 모델. **현재 가장 성능이 우수함.**
  * **목표:** `final_strategy_comparison.csv` 결과를 통해 **"왜 MLP Student를 최종 모델로 선정했는지"** 근거 확보.

### 2\. Ablation Study (구성 요소 검증)

우리가 설계한 Loss Function들이 실제로 도움이 되는지 증명합니다.
  * **확인 포인트:**
      * **w/o KL (Teacher):** 점수가 급락한다면 -\> **"Teacher의 지식 전수가 필수적임"** 증명.
      * **w/o Reg (Mono/Smooth):** 점수가 떨어진다면 -\> **"도메인 지식(시간 흐름에 따른 위험도 증가) 제약이 유효함"** 증명.

### 3\. 앙상블 및 모델 정당성 증명 (Ensemble & Justification)

\*\*"왜 Random Forest가 더 좋은데 굳이 딥러닝을 썼는가?"\*\*에 대한 방어 논리 실험입니다.
  * ** 핵심 분석 포인트 :**
      * RF/XGB는 \*\*"수치적 규칙(Rule-based)"\*\*에 강합니다. (예: 점수가 0이면 위험)
      * DL(Student)은 \*\*"의미론적 패턴(Semantic Pattern)"\*\*에 강합니다. (Teacher가 전수해준 잠재적 위험도)
      * **결론:** 두 모델은 서로 틀리는 문제가 다릅니다. 따라서 **RF + DL 앙상블 시 성능을 관찰하여, 각각의 coverage를 비교

-----

## Hyperparameters (Best Settings)

실험을 통해 도출된 최적의 파라미터는 `configs/best_params.py`에 저장되어 있습니다.

```python
# MLP Student Best Config
{
    "model_type": "mlp",
    "hidden_dim": 64,      # 32보다 64가 표현력이 좋았음
    "dropout": 0.3,        # 과적합 방지
    "lr": 0.0009,
    "focal_alpha": 0.82,   # 불균형 데이터(Risk)에 가중치 부여
    "teacher_d_token": 32  # 고정
}
```
