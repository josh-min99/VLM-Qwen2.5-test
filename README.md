# VLM(Qwen2.5-VL) vs YOLO11s — 해상 선박 탐지 정면 비교

군 경계 해상 영상에서 **VLM이 전용 탐지기를 얼마나 따라오는가**를 같은 데이터·같은 지표로 재는 실험입니다.

- 비교 대상: `github.com/josh-min99/YOLO-pipeline` 의 YOLO11s (군함 mAP50 **0.979**, 새 장소 홀드아웃)
- 이 repo: Qwen2.5-VL LoRA fine-tune + 공정 비교 평가기

---

## 🔴 가장 중요한 전제 — 데이터셋은 여기서 만들지 않습니다

이 repo에는 **데이터도 분할 파일도 없습니다.** 전부 `YOLO-pipeline`에서 만든 것을 그대로 읽습니다.

> **절대로 이 repo에서 데이터셋을 재빌드하지 마세요.**
> 분할이 1클립이라도 달라지면 val 구성이 바뀌고, 그 순간 "YOLO 0.979 vs VLM X"라는 비교가 **무효**가 됩니다. 조용히 틀리기 때문에 알아채기도 어렵습니다.

올바른 순서:

```bash
# 1) YOLO repo에서 데이터셋을 먼저 빌드 (DATASET.md 절차 그대로)
git clone https://github.com/josh-min99/YOLO-pipeline.git
cd YOLO-pipeline
#   ... DATASET.md §5 따라 datasets/marine 생성 ...
#   반드시 이 숫자가 나와야 함:
#     [train] frames=28698 boxes=29990   [val] frames=13020 boxes=14272

# 2) 이 repo는 그 결과 디렉토리를 '읽기만' 한다
cd ../VLM-Qwen2.5-test
python scripts/vlm_prepare.py --root ../YOLO-pipeline/datasets/marine --out datasets/marine_vlm
```

`vlm_prepare.py`는 라벨 JSON에서 새로 만들지 않고 **이미 빌드된 YOLO 데이터셋 디렉토리를 읽습니다.** 그게 분할 동일성을 보장하는 유일한 방법입니다.

---

## 공정 비교를 위해 맞춘 것들

VLM과 탐지기를 비교할 때 흔히 틀리는 지점 세 가지를 미리 막아뒀습니다.

| 함정 | 대응 |
|---|---|
| **해상도 불일치** — VLM에 640 정사각을 넣고 지면 VLM이 진 게 아니라 해상도가 진 것 | Qwen2.5-VL의 동적 해상도로 `max_pixels`를 1280×736에 맞춤. YOLO와 동일 |
| **좌표 공간 오해** — Qwen의 grounding 좌표는 원본이 아니라 `smart_resize` 이후 픽셀 공간 | GT를 resize 공간으로 스케일해 학습, eval에서 되돌림. `--dump-sample`로 실제 문자열 확인 |
| **지표 구현이 다름** — ultralytics mAP vs 직접 짠 mAP는 비교 불가 | 평가 코드를 **하나만** 두고 YOLO·VLM 예측을 같은 함수에 넣음 |

특히 두 번째는 틀려도 예외가 안 나고 mAP만 0 근처로 떨어져서 *"VLM은 소형 객체를 못 한다"*로 오독하기 쉽습니다.

---

## 사용법

```bash
pip install -r requirements.txt

# 0) 좌표 규약 검증 — 반드시 눈으로 확인
python scripts/vlm_prepare.py --root ../YOLO-pipeline/datasets/marine \
    --out datasets/marine_vlm --check-boxes 12
#    -> datasets/marine_vlm/check_train/*.jpg 에서 박스가 배 위에 얹혔는지 확인

# 1) 학습 전 실제 입력 문자열 확인
python scripts/vlm_train_qwen.py --data datasets/marine_vlm --dump-sample 3

# 2) LoRA fine-tune (3090 24GB 1장)
python scripts/vlm_train_qwen.py --data datasets/marine_vlm \
    --out runs/vlm/qwen25vl3b --model Qwen/Qwen2.5-VL-3B-Instruct \
    --epochs 2 --max-pixels 942080

# 3) 예측 뽑기 (두 백엔드 모두 같은 데이터로)
python scripts/vlm_eval.py predict --backend yolo --weights best_spot.pt \
    --data datasets/marine_vlm/val.jsonl --out preds_yolo.json
python scripts/vlm_eval.py predict --backend vlm \
    --adapter runs/vlm/qwen25vl3b/adapter_final \
    --data datasets/marine_vlm/val.jsonl --out preds_vlm.json

# 4) 같은 지표로 채점
python scripts/vlm_eval.py score preds_yolo.json preds_vlm.json --conf 0.6
```

---

## ⚠️ 결과 해석 시 반드시 명시할 한계

**1. 학습 데이터에 빈 프레임이 0장입니다.**
71858 개방데이터는 모든 프레임에 선박이 최소 1척 있습니다(`background=0`). VLM이 *"없다"* 고 답하는 법을 배울 기회가 없어, **빈 바다에서 환각할 위험**이 큽니다. YOLO는 구조적으로 무탐지가 자연스럽지만 VLM은 아닙니다. 이 비대칭을 결과에 적어야 합니다.

**2. VLM은 confidence를 안 뱉습니다.**
mAP는 랭킹이 있어야 정의되므로 생성 토큰의 logprob 평균을 점수로 씁니다. YOLO의 conf만큼 교정된 값이 아니므로 **헤드라인은 운영점 P/R/F1로 보고**하고 mAP는 참고로 두세요.

**3. val 전량(13,020장) 추론은 4~7시간입니다.**
생성 방식이라 장당 1~2초입니다. `--stride 4`로 줄일 수 있지만(클립 단위라 편향 적음), 그 경우 **YOLO도 같은 부분집합으로 다시 재야** 공정합니다.

**4. 이 비교는 "알려진 3클래스"에 한합니다.**
VLM의 진짜 강점은 open-set(학습에 없던 물체)인데, 이 벤치마크는 그걸 재지 않습니다. 여기서 YOLO가 이겨도 *"VLM이 쓸모없다"*가 아니라 *"닫힌 클래스 탐지에는 전용 탐지기가 낫다"*까지만 말할 수 있습니다.

---

## 참고

| | |
|---|---|
| 데이터셋 구성·분할 근거 | [YOLO-pipeline / DATASET.md](https://github.com/josh-min99/YOLO-pipeline/blob/main/DATASET.md) |
| 비교 기준 모델 | YOLO11s 1280, 군함 mAP50 0.979 (P0.951 R0.975), 추론 2.6ms |
| 이미지 원본 | AI Hub 71858 `3.개방데이터` (전부 주간·EO·C5) |
