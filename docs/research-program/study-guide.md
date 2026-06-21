# spatial-probe -- study guide (Korean)

> Moved from ~/Desktop on 2026-06-21. **Source-of-truth note:** the 6-paper summaries below are a
> 2026-06-19 first draft. Authoritative versions: `PLAN.md` (program), `research-program/` (the
> 2026-06-21 adversarial review), `experiments/<exp>/preregistration.md` (a specific run -- e.g.
> occquery's H1-headline/H3-demoted correction supersedes the OccQuery section below). What is UNIQUE
> and worth keeping here: the **glossary** (Korean terms) and the **data/access + compute/budget table**.
> Where this disagrees with PLAN.md, PLAN wins.

작성 2026-06-19. 출처: 2회의 multi-agent sweep (prior-work 전부 WebSearch 검증).
한 줄 thesis: **3D의 본질은 render가 아니라 queryable/updatable STATE.** 같은 진단 method(falsifiable physical predicate를 TEST로 써서 "representation이 행동에 필요한 신호를 실제로 STORE하는가"를 isolate)를 6개 state 축에 겨눈다.

접근성 범례: ✓ 쉬움(public data + 1 GPU eval) · △ 손 많이 감/예산 필요 · 내 도구 = voxel engine·SplatCarve(남이 없는 부분).

---

## 데이터/인프라 접근 요약 (먼저)

| # | 주제 | 데이터 (전부 public) | compute | 접근성 |
|---|---|---|---|---|
| 1 | GT-distrust | CARLA, Occ3D-nuScenes/Waymo | eval, 1 GPU | ✓ |
| 2 | Value-of-Correction | Occ3D, nuScenes, CARLA/UniOcc | **retrain 루프, multi-GPU-day** | △ 예산 (proxy로 완화) |
| 3 | Visibility Calibration | SemanticKITTI/SSCBench, CARLA + 공개 weight | **eval-only, 1 GPU** | ✓ 제일 쉬움 |
| 4 | OccQuery | Occ3D, Argoverse2, SSCBench | eval(GT), 1 GPU | ✓ + 내 voxel engine |
| 5 | ASOF | Occ3D, LiDAR+HD-map, DTU + 변환기 | GS 재구성 ×N, sub-hour/scene 1 GPU | △ 손 많이 감 |
| 6 | DynField-Necessity | Cam4DOcc, nuPlan/WOMD, Occ3D | open-loop 1 GPU / closed-loop 무거움 | △ open-loop만 v1 |

**공통 현실:** 모든 데이터셋은 무료(단 Waymo/nuScenes/Argoverse는 research 계정 + 약관 동의 필요). 대부분 "첫 실험"은 eval-only / single GPU라 RTX 4090급 또는 cloud burst(vast.ai/runpod/lambda, A100 ~$1-2/hr)로 충분. retrain 무거운 건 2번·6번 closed-loop뿐 → proxy/small model로 de-risk. **내가 이미 가진 voxel engine + SplatCarve가 1·4·5의 핵심 엔진이라 남이 못 따라오는 부분.**

---

## 1. GT-distrust — occlusion 기하로 GT label-error 예측 (visibility 축, occlusion line)

- **가설.** accumulated LiDAR 위로 DDA line-of-sight를 통과시켜 얻은 per-voxel "occlusion-depth"(이 voxel이 occluder를 통과해서만 지지되는 정도)가, 어떤 occupancy GT voxel이 *틀렸는지*를 모델 softmax confidence나 ReliOcc/OCCUQ uncertainty보다 잘 예측한다.
- **핵심 아이디어.** occlusion을 (남들처럼) supervision 억제나 모델 uncertainty가 아니라 **GT 라벨 신뢰성 audit predictor**로 사용. 내 논문의 inverse — 모델 품질이 아니라 라벨 품질에 occlusion-as-test를 적용.
- **측정 기준.** label-distrust **AUROC/AUPR** (clean/amodal GT 대비) · review-trigger precision/recall · 고정 recall에서 review-minutes-saved.
- **데이터/접근.** CARLA/Co3SOP(exact amodal GT라 "label error"가 모호하지 않음 = falsifiable core) + Occ3D-nuScenes/Waymo(visibility mask). 전부 public, eval 위주 single GPU. ✓ 접근 쉬움. real-LiDAR는 amodal truth가 없어 stretch cross-check으로만.

## 2. Value-of-Correction — occluded label 고치면 metric 얼마나 움직이나 (occlusion line)

- **가설.** occlusion-conditioned value-of-correction estimator가 "occluded label을 고치면 occluded-slice metric을 가장 많이 올리는 scene"을 LMD/uncertainty/ActiveAD/random보다 잘 랭크한다.
- **핵심 아이디어.** PRISM headline moat("뭐가 mislabeled + 그게 모델 성능에 어떻게 영향")를 scoring function 하나로. Shah et al.(2310.02533)의 slice-targeted value-of-correction을 occupancy/occlusion/AV로 처음 instantiate(이건 인용 필수, 내 건 3D/AV instantiation + benchmark).
- **측정 기준.** top-K 교정 → retrain → **occluded-slice Δ-mIoU / Δ-RayIoU**(headline) · predicted value ↔ realized gain **Spearman rank-corr** · gain-per-review-minute. fairness: content-sufficient control slice의 gain이 occlusion 신호로 설명되면 안 됨.
- **데이터/접근.** Occ3D-nuScenes + native nuScenes visibility token + CARLA/UniOcc. **단 correct→retrain→measure 루프가 multi-GPU-day라 조합 폭발.** de-risk 필수: predicted value는 proxy/1-run attribution(TracIn/Data-Shapley-in-one-run), causal 검증은 small proxy occupancy 모델 + 주입한 known-GT corruption subset. △ 예산 필요.

## 3. Visibility-Conditional Calibration — uncertainty가 "못 보는 곳"에서 정직한가 (occlusion line)

- **가설.** aggregate ECE는 occluded-region overconfidence를 숨긴다. visibility로 stratify한 **vc-ECE / occluded-coverage-gap**이 그걸 드러내고, visibility-agnostic confidence threshold보다 나은 자동 review trigger다.
- **핵심 아이디어.** 같은 occlusion-as-test를 uncertainty 축에 — "모델이 *볼 수 없는 바로 그곳에서* confidence가 정직한가". PaSCo(이미 occupancy ECE + empty/non-empty 분해)와 MapDiffusion(visible vs invisible uncertainty *magnitude*) 사이를 "calibration READ + fairness control"로 파고듦.
- **측정 기준.** stratum별(observed / behind-occluder / out-of-FOV) **vc-ECE, MCE, reliability diagram** · headline = **occluded-coverage-gap**(0.90 − behind-occluder 실측 conformal coverage) · occluded stratum 내 misclassification AUROC. fairness: visible-sufficient control은 near-calibrated여야.
- **데이터/접근.** SemanticKITTI / SSCBench-KITTI-360(amodal GT 기본) + CARLA/Co3SOP. **eval-only, 공개 checkpoint(PaSCo/MonoScene), 새 코드 수백 줄, single GPU, 주 단위.** ✓ 접근 제일 쉬움.

## 4. OccQuery — occupancy-native physical-predicate retrieval (geometry/occupancy 축, non-occ 1순위)

- **가설.** clearance / free-corridor / reachable-region 같은 deterministic geometric predicate를 dense occupancy field 위에서 실행하면, box-only 언어(RefAV 28함수)가 *표현조차 못 하는* planning-critical scene을 retrieve할 수 있고, occupancy oracle 대비 denotation-correct하게 유지된다.
- **핵심 아이디어.** physical-predicate retrieval을 object cuboid에서 떼어내 voxel/occupancy 위로. (Scenic이 visibility predicate는 선점 → 거긴 demote, **clearance/free-space로 리드.**) PRISM의 "physical-quantity로 직접 측정하는 reproducible search"를 그들이 *이미 만드는* occupancy asset 위에서 확장.
- **측정 기준.** denotation accuracy(occupancy oracle 대비 precision/recall/**F1**) · 연속값(clearance)의 **Tolerance-based Accuracy**([75%,125%] of GT) + MAE · **query-expressibility coverage**(occupancy로 답 가능 vs cuboid로 표현 불가 개수) · predicted-occupancy noise → retrieval 붕괴 곡선.
- **데이터/접근.** Occ3D-nuScenes/Waymo + SSCBench(occupancy GT) + Argoverse2 LiDAR/HD-map(독립 free-space/clearance GT); RefAV cuboid predicate = expressivity-gap baseline. **GT-occupancy면 eval만, single GPU. ✓ + 내 voxel engine이 곧 predicate executor.**

## 5. ASOF — render→state 변환 충분성 (conversion 축, non-occ 2순위)

- **가설.** render→state 변환(3DGS/NeRF/mesh → occupancy)이 photometric quality(PSNR/LPIPS)는 유지하면서 행동에 필요한 신호(free-along-trajectory, distance-within-margin 등)는 떨어뜨릴 수 있다. ASOF가 Chamfer/IoU/RayIoU와 *다르게* 변환기를 재정렬하고 PSNR과 decorrelated하다.
- **핵심 아이디어.** detector가 아니라 **변환기 자체**를 action-predicate correctness로 채점 + **matched-render control**(render 품질 고정한 채 thin structure를 carve → sufficiency 붕괴) — 내 text-sufficient fairness control의 직계 후손. SplatCarve가 곧 controllable 변환기(carve 강도/opacity/해상도 sweep).
- **측정 기준.** ASOF(query distribution 위 action-predicate correctness) · reachable-set 내 false-free volume(**TTC**로 weight) · ASOF vs Chamfer/voxel-IoU/RayIoU **rank-correlation**(재정렬 입증) + vs PSNR/LPIPS(blindness 입증) · **matched-render control separation**(품질 고정 시 sufficiency만 0.0↔1.0).
- **데이터/접근.** Occ3D(occupancy GT + RayIoU harness) + raw nuScenes/Waymo/KITTI-360 LiDAR+HD-map(oracle: stop line, ground plane) + DTU/Tanks&Temples(surface contrast). 변환기: 2DGS→mesh→voxel, Gaussian Opacity Fields, TSDF/VDBFusion, + SplatCarve. GS 재구성이 scene당 sub-hour 1 GPU이나 변환기 여러 개 × scene 여러 개라 손 많이 감. △.

## 6. DynField-Necessity — 어떤 dynamics field를 planner가 실제로 필요로 하나 (dynamics/시간 축, non-occ 3순위)

- **가설.** 저장된 dynamics field(per-cell velocity, instance flow, accel, heading-rate)를 하나씩 corrupt하면 *aggregate* downstream-plan delta는 ~0(known null)이지만, **accel/cut-in/low-TTC regime에서는 크게 튀고**, intrinsic 품질(flow-EPE/mIoU@horizon)은 그 necessity를 예측하지 못한다.
- **핵심 아이디어.** occlusion method를 시간축으로 직접 port — falsifiable motion predicate(|a|>τ, turning, TTC<τ, cut-in)를 test로 쓰고 단일 stored dynamics FIELD를 ablate해 downstream necessity를 isolate. **⚠ regime-tail이 garnish가 아니라 실험 본체**(aggregate가 0이면 known null 재현이라, novelty 전체가 "necessity가 tail에 몰린다 + intrinsic이 예측 못 한다"에 걸림).
- **측정 기준.** per-field downstream-delta(plan L2 / collision-rate / closed-loop score 변화) = **necessity** · **regime × field necessity table** · per-frame flow-EPE와 plan-delta의 **Spearman ρ**(intrinsic이 necessity 예측 못 함을 입증, ρ≈0이 지지 신호).
- **데이터/접근.** Cam4DOcc(nuScenes-Occupancy + 3D flow + instance) + Argoverse2 4D-occupancy + nuPlan/WOMD(closed-loop) + Occ3D. planner는 **frozen/off-the-shelf 중 occupancy+flow를 input으로 소비하는 것**(flow를 무시하는 planner 쓰면 null이 trivial). open-loop는 single GPU 가능, closed-loop(nuPlan)는 무겁고 brittle → v1 open-loop. △.

---

## 용어집 (공부용)

**표현/데이터**
- **voxel** — volume+pixel, 3D 픽셀. 공간을 정육면체 격자로 자른 한 칸.
- **occupancy / occupancy grid** — 각 voxel이 occupied / free / unknown인지의 격자. class를 몰라도 "공간이 찼나"를 앎.
- **occupancy flow / scene flow** — 각 voxel/점이 다음 순간 어디로 움직이는지의 벡터장.
- **amodal GT** — 가려진 부분까지 포함한 "진짜" 정답. sim(CARLA)·SSC 데이터에서만 정확히 알 수 있음(real LiDAR엔 없음).
- **HD map / static infra** — 차선/정지선/도로 구조 등 정적 지도 레이어.
- **VRU** — vulnerable road user(보행자·자전거 등).

**기하 연산 (내 도구)**
- **DDA / raycast / line-of-sight** — 격자를 따라 ray를 한 칸씩 진행시키며 충돌(solid voxel) 검사. 내 occlusion 0.982가 여기서 나옴.
- **clearance** — 가장 가까운 occupied voxel까지의 거리(여유 간격).
- **free corridor / reachable region** — 충돌 없이 지나갈 수 있는 빈 통로 / flood-fill로 도달 가능한 자유 공간.
- **Chamfer distance** — 두 표면/점군 사이 거리. surface fidelity 표준 metric.
- **TTC (time-to-collision)** — 충돌까지 남은 시간. 안전 가중치로 씀.

**평가 metric**
- **mIoU** — mean Intersection-over-Union, 예측∩정답 / 예측∪정답의 평균. occupancy/seg 표준 정확도.
- **RayIoU** — ray 기반 occupancy IoU(두꺼운 표면 과보상 문제를 고침).
- **AUROC / AUPR** — 이진 판별기("이 라벨 틀렸나")의 ROC/PR 곡선 아래 면적. 1=완벽, 0.5=랜덤.
- **ECE (Expected Calibration Error)** — "90% 확신"이라 할 때 실제 90% 맞는지의 평균 괴리. **calibration** = 확신과 실제 정확도의 일치.
- **conformal coverage / set size** — 예측 집합이 정답을 포함하는 비율(목표 0.90) / 그 집합 크기.
- **denotation accuracy** — query(프로그램)를 실행해 나온 *결과 집합*이 정답 집합과 맞는지(NL→code 평가법). denotation = 표현이 가리키는 실제 결과.
- **Tolerance-based Accuracy** — 연속값 예측이 GT의 [75%,125%] 안이면 정답으로 치는 관대 정확도.
- **PSNR / LPIPS** — 이미지 품질. PSNR=픽셀 차이(클수록 좋음), LPIPS=지각적 차이(작을수록 좋음).
- **HOTA** — tracking/scenario-mining 종합 metric(RefAV가 씀).

**개념/방법**
- **predicate** — 참/거짓을 반환하는 조건 함수("clearance < 0.5m?"). **physical predicate** = 물리량으로 정의된 술어.
- **oracle** — 완벽한 입력/정답을 가정한 상한 측정(perception을 분리하려 GT를 넣음).
- **fairness control** — 내 핵심 무기. "반대로 쉬운 query"(content/render로 답되는)도 같이 재서, 도구가 한쪽 편든 게 아님을 증명. 내 논문의 text-sufficient control.
- **ablation** — 한 요소를 빼거나 망가뜨려 그게 결과에 얼마나 필요한지 보는 실험.
- **regime** — 상황 구간(등속 / 가감속 / 회전 / 끼어들기 / 저-TTC).
- **planner / open-loop vs closed-loop** — planner=주행 계획 모듈. open-loop=한 스텝 예측을 정답과 비교(가벼움). closed-loop=실제로 굴려 충돌까지 봄(무겁고 brittle).
- **value-of-correction / data valuation / influence / Data Shapley / TracIn** — 어떤 데이터(또는 라벨 수정)가 모델 성능에 기여하는 정도를 추정하는 계열.
- **provenance / lineage** — 데이터의 출처·이력(어느 센서/버전/시점이 만들었나). 재현·audit·revocation에 필요.
- **scene / scenario mining (retrieval)** — 대량 주행 로그에서 조건 맞는 장면을 찾아내는 것.
- **SSC (Semantic Scene Completion)** — 부분 관측에서 가려진 곳까지 포함한 dense 3D를 채우는 task(amodal GT 기본).

**주요 데이터셋**
- **nuScenes / Waymo Open / Argoverse2** — 대표 AV multi-sensor 데이터셋(무료, research 계정).
- **Occ3D-nuScenes / Occ3D-Waymo** — 위 데이터에 visibility-aware occupancy 라벨을 입힌 occupancy benchmark.
- **SemanticKITTI / SSCBench-KITTI-360** — SSC 데이터(amodal GT 기본).
- **Cam4DOcc** — 4D(시간) occupancy + flow benchmark.
- **CARLA** — 오픈소스 주행 시뮬레이터(exact GT를 마음대로 뽑음 = falsifiable core용). **Co3SOP / UniOcc** — CARLA 등 기반 occupancy benchmark.
- **nuPlan / WOMD(WOSAC)** — closed-loop planning/simulation benchmark.
- **DTU / Tanks&Temples** — 3D 재구성 품질(surface) 표준 데이터.
- **RefAV** — Argoverse2 위 NL scenario-mining(28개 cuboid 함수) = OccQuery의 box-only baseline.
