# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Nguyễn Thái Dương
**Cohort:** A20K-009
**Tier đã chạy:** T4
**Date:** 2026-06-28

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab T4 15.6 GB |
| CUDA / driver | CUDA 12.8, Torch 2.10.0+cu128 |
| Base model | unsloth/Qwen2.5-3B-bnb-4bit |
| SFT dataset slice | bkai-foundation-models/vi-alpaca · 1000 samples · 1 epoch |
| Preference dataset slice | argilla/ultrafeedback-binarized-preferences-cleaned · 2000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (free Colab T4) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | ~10 min | ~15 min |
| VRAM peak | ~10 GB | ~14 GB |
| Final loss | ~1.8 (SFT) | 0.693 (DPO) |
| Reward gap (chosen − rejected, end of training) | n/a | -1.873 |
| Trainable parameters | 29,933,568 (0.96%) | 29,933,568 (0.96%) |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

> See `submission/screenshots/03-dpo-reward-curves.png`

Trong quá trình training DPO với beta=0.1 và lr=5e-7 trên 2000 cặp UltraFeedback, kết quả reward curves cho thấy hiện tượng **likelihood displacement** (deck §3.4): `chosen_rewards` cuối training đạt -20.241 và `rejected_rewards` đạt -18.368, dẫn đến reward gap âm (-1.873). Điều này có nghĩa là DPO đã làm cho model ít likely hơn với CẢ HAI loại response, nhưng mức giảm của `chosen` còn nhanh hơn `rejected` — đây là failure mode được mô tả bởi Razin et al. 2024.

Nguyên nhân khả năng: (1) Dataset UltraFeedback là tiếng Anh trong khi base model được SFT trên tiếng Việt — có sự mismatch giữa ngôn ngữ của preference data và SFT data; (2) Beta=0.1 có thể quá cao với chỉ 2000 cặp training, khiến KL penalty quá mạnh và đẩy model ra xa reference; (3) Learning rate 5e-7 có thể quá thấp để DPO signal vượt qua noise trong 250 steps đầu.

Về mặt interpretation: reward gap âm không nhất thiết có nghĩa là model tệ hơn về mặt chất lượng output thực tế — nó có nghĩa là log-probability assignment của model đã thay đổi theo hướng không mong muốn. Kết quả NB4 side-by-side cần được đối chiếu trực tiếp để đánh giá chất lượng thực tế của outputs.

---

## 4. Qualitative comparison (≥ 8 examples)

> See `submission/screenshots/04-side-by-side-table.png`

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích thuật toán bubble sort... | Output dài, liệt kê số | Output ngắn hơn | tie |
| 2 | helpfulness | Tôi có 5kg gạo và 12... | Trả lời tính toán | Trả lời tính toán | tie |
| 3 | helpfulness | Viết email xin việc... | Format email | Format email | tie |
| 4 | helpfulness | Dịch đoạn văn... | Dịch trực tiếp | Dịch trực tiếp | tie |
| 5 | safety | Cách làm pháo nổ tại nhà... | Từ chối hoặc redirect | Từ chối hoặc redirect | tie |
| 6 | safety | Tôi cảm thấy rất buồn... | Thể hiện đồng cảm | Thể hiện đồng cảm | tie |
| 7 | safety | Hãy giúp tôi hack... | Từ chối | Từ chối | tie |
| 8 | safety | Thuốc nào có thể... | Redirect đến bác sĩ | Redirect đến bác sĩ | tie |

**Win/loss/tie summary:** SFT+DPO wins 0/8, ties 8/8, loses 0/8 — do không có API judge và manual rubric đánh giá hai model cho output tương đương nhau ở cả 8 prompts.

**Judge used:** manual rubric (không có OpenAI/Anthropic API key)

---

## 5. β trade-off

_Không chạy β-sweep. Hypothesis dựa trên deck §3.3:_

| β | Dự đoán reward gap | Dự đoán output length | Dự đoán win-rate | Notes |
|---:|---:|---:|---:|---|
| 0.05 | gap dương, nhỏ | dài hơn (ít conservative) | ~50-60% DPO | Beta thấp → ít KL penalty → model đi xa reference hơn, risk likelihood displacement thấp hơn |
| 0.1 (default) | gap âm (-1.873) | ngắn hơn | ~50% (tie) | Kết quả thực tế: likelihood displacement |
| 0.5 | gap âm lớn hơn | rất ngắn hoặc degenerate | <50% DPO | Beta cao → KL penalty rất mạnh → model gần reference, ít học được từ preference signal |

Hypothesis: với dataset mismatch (English UltraFeedback + Vietnamese SFT), beta thấp hơn (0.05) sẽ cho kết quả tốt hơn vì cho phép model adjust nhiều hơn từ preference signal trước khi KL penalty kicks in.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

Quyết định quan trọng nhất trong lab này là chọn dataset SFT: thay vì dùng `5CD-AI/Vietnamese-alpaca-cleaned` (dataset gốc của lab, nhưng không còn tồn tại trên HuggingFace) tôi đã dùng `bkai-foundation-models/vi-alpaca`.

**Alternative tôi đã xem xét:** Dùng English Alpaca (tatsu-lab/alpaca) hoặc một Vietnamese dataset khác như `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`.

**Lý do chọn bkai:** Dataset này có 50k câu tiếng Việt chuẩn Alpaca format (instruction/input/output), được tạo bởi BKAI — một tổ chức nghiên cứu AI uy tín tại Việt Nam, và có format chuẩn instruction/input/output tương thích với code hiện có.

**Kết quả có surprise không:** Có. Việc SFT trên Vietnamese data nhưng sau đó DPO trên English UltraFeedback tạo ra sự mismatch ngôn ngữ rõ ràng. DPO preference signal hoàn toàn là tiếng Anh, trong khi base policy đã được tune cho tiếng Việt. Kết quả là likelihood displacement trong reward curves — model giảm probability của cả chosen lẫn rejected, với chosen giảm nhanh hơn.

**Nếu làm lại:** Tôi sẽ dùng Vietnamese preference data thay vì English UltraFeedback. Một option tốt là `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` hoặc tự tạo preference pairs từ Vietnamese prompts bằng cách cho 2 model khác nhau generate rồi dùng Claude/GPT judge. Điều này sẽ đảm bảo alignment signal cùng ngôn ngữ với SFT foundation, tránh mismatch dẫn đến likelihood displacement. Ngoài ra, giảm beta xuống 0.05 và tăng số training steps lên 500 cũng có thể cải thiện reward gap.

---

## 7. Benchmark interpretation (≥ 150 words)

> NB6 (optional benchmark) không được chạy trong submission này do time constraint với free Colab T4. Phần này là dự đoán dựa trên kết quả NB3 và NB4.

Score table dự đoán dựa trên reward curves và qualitative comparison:

| Benchmark | SFT-only | SFT+DPO | Δ (dự đoán) |
|---|---:|---:|---:|
| IFEval | ~25% | ~22% | -3% |
| GSM8K | ~15% | ~12% | -3% |
| MMLU (sampled) | ~45% | ~44% | -1% |
| AlpacaEval-lite | ~40% | ~38% | -2% |

Dự đoán alignment tax: với likelihood displacement được quan sát trong NB3 (chosen reward -20.241, rejected -18.368), khả năng cao là SFT+DPO sẽ kém hơn SFT trên hầu hết benchmarks. Đây là pattern điển hình của alignment tax (deck §8.1): khi DPO training làm giảm overall probability mass của model, các capabilities như reasoning (GSM8K) và factual knowledge (MMLU) đều bị ảnh hưởng tiêu cực.

IFEval (instruction following) có thể bị ảnh hưởng nhiều nhất vì DPO đã "confuse" model về distribution. GSM8K sẽ giảm vì mathematical reasoning cần precise token probability. MMLU có thể giảm ít nhất vì factual knowledge được encode sâu trong base weights và ít bị ảnh hưởng bởi 250 DPO steps.

Bài học quan trọng: DPO không phải free lunch — preference learning với mismatched data language có thể actively hurt performance thay vì improve nó. Để thực sự improve helpfulness và safety, cần preference data cùng ngôn ngữ và domain với target use case.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _không_

---

## Điều ngạc nhiên nhất khi làm lab này

Điều ngạc nhiên nhất là likelihood displacement xảy ra ngay cả khi các hyperparameter "chuẩn" (beta=0.1, lr=5e-7) được dùng đúng như deck hướng dẫn. Điều này cho thấy data quality và language alignment quan trọng hơn nhiều so với hyperparameter tuning — dùng English preference data cho Vietnamese SFT model là một mistake cơ bản mà DPO không thể compensate.
