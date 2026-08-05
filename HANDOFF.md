# 인계 — 학습된 Qwen2.5-VL 어댑터로 평가하기

> 이 문서는 **모델을 받아서 YOLO와 비교 평가를 진행할 사람**을 위한 것입니다.
> 학습은 끝났고, 여기서는 예측을 뽑아 채점하는 것만 하면 됩니다.

**받을 것은 두 개뿐입니다** (Drive 공유폴더 `vlm_handoff/`):

| 파일 | 크기 | 내용 |
|---|---|---|
| `qwen25vl3b_adapter.tgz` | 265MB | 학습된 Qwen2.5-VL LoRA 어댑터 |
| `best_spot.pt` | 19MB | 비교 기준 YOLO11s |

> ⚠️ 같은 Drive에 `71856_train/`(합성데이터 77GB)이 있지만 **이 실험과 무관합니다.**
> VLM도 YOLO 기준모델도 **71858 실데이터만** 학습했습니다. 그쪽은 보지 마세요.

---

## 🔴 어댑터만으로는 못 씁니다

산출물은 **LoRA 어댑터**(전체 모델이 아님)입니다. 돌리려면 네 가지가 다 맞아야 합니다.

| 필요한 것 | 값 |
|---|---|
| ① 어댑터 | `runs/vlm/qwen25vl3b/adapter_final/` |
| ② 베이스 모델 | `Qwen/Qwen2.5-VL-3B-Instruct` |
| ③ 입력 해상도 | `max_pixels = 942080` (= 1280×736) |
| ④ **좌표 규약** | 예측 좌표는 **`smart_resize` 이후 공간** — 원본으로 되돌려야 함 |

**④가 가장 위험합니다.** 되돌리지 않으면 박스가 전부 어긋나 mAP가 0 근처로 나오고, *"VLM은 소형 객체를 못 한다"*는 잘못된 결론에 도달합니다. `vlm_eval.py`가 이 변환을 자동으로 처리하므로 **그 스크립트를 쓰는 것이 가장 안전합니다.**

---

## 절차

```bash
# 0) 환경
git clone https://github.com/josh-min99/VLM-Qwen2.5-test.git
cd VLM-Qwen2.5-test && pip install -r requirements.txt
#    설치 후 torch.cuda.is_available() 과 get_device_name(0) 을 반드시 같이 확인

# 1) 데이터셋 — 이 repo에서 만들지 말 것 (README 참고)
#    YOLO-pipeline에서 datasets/marine 을 먼저 빌드하고 그 결과를 읽는다
python scripts/vlm_prepare.py --root ../YOLO-pipeline/datasets/marine --out datasets/marine_vlm
#    -> [val] 13020 images / 14272 boxes  이어야 함. 다르면 벤치마크가 다른 것

# 2) 어댑터·기준모델 받기 — Drive 공유폴더의 vlm_handoff/ 안에 둘 다 있음
#      qwen25vl3b_adapter.tgz  265MB   학습된 VLM 어댑터
#      best_spot.pt             19MB   비교 기준 YOLO11s (실데이터만 학습)
mkdir -p runs/vlm/qwen25vl3b && tar xzf qwen25vl3b_adapter.tgz -C runs/vlm/qwen25vl3b
#    -> runs/vlm/qwen25vl3b/adapter_final/ 안에 USAGE.txt 도 들어 있음

# 3) 예측 뽑기 (두 백엔드 모두 같은 데이터로)
python scripts/vlm_eval.py predict --backend yolo --weights best_spot.pt \
    --data datasets/marine_vlm/val.jsonl --out preds_yolo.json
python scripts/vlm_eval.py predict --backend vlm \
    --adapter runs/vlm/qwen25vl3b/adapter_final \
    --base-model Qwen/Qwen2.5-VL-3B-Instruct \
    --data datasets/marine_vlm/val.jsonl --out preds_vlm.json
#    🔴 인자명은 --base-model 이다(--model 아님).
#    베이스 모델을 미리 받아뒀다면 로컬 경로도 됨: --base-model /path/to/qwen25vl3b

# 4) 같은 지표로 채점
python scripts/vlm_eval.py score preds_yolo.json preds_vlm.json --conf 0.6
```

**YOLO 쪽 가중치는 반드시 `best_spot.pt`** (실데이터만 학습). W8 계열은 합성이 섞여 있어 비교가 기울어집니다 — README 첫 절 참고.

---

## 시간이 오래 걸립니다

**실측 장당 2.8초**(RTX 3090 1장, 스모크 10장 p50). val 전량 13,020장이면 **약 10시간**입니다. 줄이려면:

```bash
python scripts/vlm_prepare.py --root ... --out datasets/marine_vlm --stride 4
```
클립 단위 stride라 편향이 적습니다. **단, 그 경우 YOLO도 같은 부분집합으로 다시 재야** 공정합니다.

---

## 결과 보고 시 반드시 명시할 것

1. **VLM은 confidence를 안 뱉습니다.** mAP는 랭킹이 있어야 정의되므로 생성 토큰의 logprob 평균을 점수로 씁니다. YOLO의 conf만큼 교정된 값이 아니므로 **헤드라인은 운용점 P/R/F1로** 하고 mAP는 참고로 두세요.
2. **학습셋에 빈 프레임이 0장입니다.** 71858 개방분은 모든 프레임에 선박이 있어 VLM이 "없다"고 답하는 법을 못 배웠습니다. **빈 바다에서 환각 위험**이 있고, YOLO는 구조적으로 무탐지가 자연스러운 반면 VLM은 아닙니다. 이 비대칭을 적어야 합니다.
3. **YOLO에 유리한 조건이 하나 있습니다.** `best_spot.pt`는 val을 조기 종료·체크포인트 선택에 썼고, VLM은 2 epoch 고정이라 그 이점이 없습니다.
4. **이 비교는 닫힌 3클래스 한정입니다.** VLM의 강점인 open-set(학습에 없던 물체)은 이 벤치마크가 측정하지 않습니다. YOLO가 이겨도 *"VLM이 쓸모없다"*가 아니라 *"닫힌 클래스 탐지에는 전용 탐지기가 낫다"*까지만 말할 수 있습니다.

---

## 인계 전 검증한 것 (이쪽에서 확인 완료)

| 항목 | 결과 |
|---|---|
| 어댑터 로드 + 예측 | ✅ 정상 |
| 파싱 실패율 | **0%** (10장) |
| 좌표 정확성 | ✅ 연속 프레임에서 박스가 부드럽게 이동 (1591→1518→1456→1391) |
| 추론 지연 | 장당 2.77초 (p50) |

즉 **모델과 좌표 변환은 이미 검증됐습니다.** 남은 건 전량 추론과 채점입니다.

---

## 기준값 (YOLO11s, `best_spot.pt`)

| | 값 |
|---|---|
| 군함 mAP50 | **0.979** (P 0.951 / R 0.975) |
| 전체 mAP50 | 0.961 |
| 추론 | 2.6 ms/장 |
| 평가셋 | 71858 지점 홀드아웃 val 13,020장 / 14,272박스 |

## 학습 설정 (재현용)

```
모델      Qwen/Qwen2.5-VL-3B-Instruct
방식      LoRA r=32, ViT 동결, 학습 파라미터 74.3M / 3.83B (1.94%)
해상도    max_pixels 942080 -> 1920×1080 이미지가 1288×700로 리사이즈
배치      per_device 1 × 4 GPU × grad_accum 8 = 유효 32
epoch     2, lr 1e-4 cosine, bf16, gradient checkpointing
데이터    71858 실데이터 train 28,698장 (합성 미포함)
하드웨어  RTX 3090 ×4, torchrun DDP
```
