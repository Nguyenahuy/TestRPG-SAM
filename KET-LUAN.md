# Kiểm chứng ba khoảng trống của RPG-SAM — phân tích và kết quả

Tài liệu này trả lời hai câu hỏi: (1) kế hoạch trong `ke-hoach-kiem-chung-RPG-SAM.md` có
kiểm chứng đúng thứ cần kiểm chứng không, và (2) chạy xong thì ba khoảng trống A/B/C có
đứng vững không.

Nền tảng: bản **replication** RPG-SAM dựng từ công thức (tác giả chưa release code), đạt
76.47 mIoU / 83.36 mDice trên Kvasir-SEG so với 78.65 / 85.65 công bố. Toàn bộ số dưới đây
nói về bản replication này. Chi tiết sinh tự động ở
[`outputs/analysis/GAPS.md`](outputs/analysis/GAPS.md) (Phase 1–2) và
[`outputs/analysis/GAP_REPORT.md`](outputs/analysis/GAP_REPORT.md) (Phase 3).

---

## Phần 1 — Kế hoạch có lỗ hổng gì

Kế hoạch đặt đúng ba câu hỏi, nhưng **bốn tiêu chí kết luận của nó không phân biệt được
điều cần phân biệt**. Đây là các chỗ đã phải sửa trước khi chạy:

### 1.1. B1: tiêu chí đúng nhưng không đặc hiệu cho RPG-SAM

> *"Recall nhóm nhỏ thấp hơn nhóm lớn ≥10 điểm % → gap xác nhận"*

Tiêu chí này gần như chắc chắn đúng với **mọi** phương pháp phân đoạn polyp — vật thể nhỏ
thì khó hơn, đó là hằng số của bài toán chứ không phải khuyết điểm của RPG-SAM. Xác nhận
nó xong vẫn không biết RPG-SAM có tệ hơn baseline hay không.

**Đã sửa:** mọi phân tầng đều tính song song cho RPG-SAM, OP-SAM và PerSAM trên **cùng
ảnh support, cùng tập query**. Tiêu chí trở thành *tương đối*: khoảng cách nhỏ–lớn của
RPG-SAM có rộng hơn của baseline không?

### 1.2. C2: tiêu chí giả định sai cơ chế

> *"Nhóm solidity(GT) thấp có `S_geo` và IoU thấp hơn rõ rệt"*

Kế hoạch giả định hai thứ đi cùng chiều. Đo thực tế thì **ngược**: `S_geo` tính trên chính
GT *giảm* khi solidity *tăng* (0.730 → 0.407 từ Q1 đến Q4), vì số hạng Scale Consensus lấn
át số hạng solidity, và solidity lại tương quan âm với kích thước (Q1 có polyp lớn nhất
22.3%, Q4 nhỏ nhất 7.7%).

Nếu chỉ theo kế hoạch, ta sẽ thấy IoU nhóm solidity thấp kém hơn (đúng), rồi **quy sai
nguyên nhân cho Weighted Solidity** trong khi thủ phạm là Scale Consensus.

**Đã sửa:** báo cáo `S_geo(GT)` và `scale(GT)` tách riêng theo từng tứ phân vị, và bảng
chéo kích thước × solidity (Phase 4) để gỡ nhiễu.

### 1.3. A2: tiêu chí gần như không thể bác bỏ

> *"Tương quan dương mạnh (r > 0.7) → gap xác nhận"*

Mask cuối được sinh ra **từ** `M_prior` (PIR chọn mask có IoU cao nhất so với prior), nên
tương quan cao là hệ quả kiến trúc, đo được trước khi chạy. Hơn nữa r cao còn có thể chỉ
phản ánh "ảnh dễ thì cả prior lẫn final đều tốt".

**Đã sửa:** thêm phân tích **recovery** = `(IoU_final − IoU_prior) / (1 − IoU_prior)` chia
theo dải chất lượng prior. Đây mới là dạng bác bỏ được: nếu PIR sửa được lỗi thượng nguồn
thì recovery phải **cao nhất** ở dải prior kém.

### 1.4. B3 không chạy được như mô tả

Kế hoạch yêu cầu chạy Kvasir-H và Piccolo rồi so với số công bố của OP-SAM (57.31 / 65.48).
Cả hai bộ đều sau đăng ký, **và** so sánh với số công bố trên split khác là không hợp lệ.
**Đã sửa:** dùng subset kích thước cực đoan của Kvasir làm vật thay thế, và so với
OP-SAM **chạy lại trên đúng split đó**, có ghi rõ đây không phải Kvasir-H thật.

### 1.5. Kế hoạch bỏ sót phê phán mạnh nhất về C

Ba thí nghiệm C1–C4 đều là thực nghiệm. Nhưng phê phán chắc chắn nhất về `S_geo` là
**suy ra từ chính công thức**, không phụ thuộc cách triển khai. Nguyên văn Section 2.2:

> `S_geo(M) = Σᵢ (|Cᵢ|/|M|)·(|Cᵢ|/|Hull(Cᵢ)|) · min(1, |M|/A_ref)`
>
> *"This term downweights candidates that are significantly **smaller** than a reference
> area `A_ref`, which represents the expected median scale of polyps."*

Số hạng Scale Consensus **một phía theo đúng thiết kế**: chỉ phạt mask nhỏ hơn `A_ref`,
không có bất kỳ số hạng nào phạt mask lớn hơn. Vì diện tích vùng vượt ngưỡng giảm đơn điệu
khi τ tăng, `S_geo` gần như đơn điệu giảm theo τ, nên GAS thoái hoá thành "lấy đáy dải
quét" — 90.8% số ảnh trong bản replication này. Điều đó đúng với *mọi* heatmap.

**Thêm một vấn đề nữa lộ ra khi đối chiếu nguyên văn:** bài định nghĩa `A_ref` là *"expected
median scale of polyps"*. Trung vị GT đo được trên Kvasir là **11.40%** diện tích ảnh,
nhưng `A_ref` mà bản replication dùng (diện tích polyp trong ảnh support) là **17.11%** —
lệch 50%. Bài **không nói** cách tính trung vị đó trong bối cảnh one-shot, nơi chỉ có duy
nhất một mask được chú thích. Đây là chỗ under-specification, không phải lỗi triển khai.

**Đã thêm:** biến thể `sym_scale` (`min(A/A_ref, A_ref/A)`) trả lại vế phạt còn thiếu,
`perim` thay solidity diện tích bằng perimeter convexity, và các biến thể `A_ref` cố định
5%/10% — trong đó 10% sát với "median scale" mà bài tuyên bố hơn là diện tích polyp support.

### 1.6. Phase 0.5 chưa đạt đúng ngưỡng kế hoạch đặt ra

Kế hoạch đòi sai lệch ±1–2 điểm. Thực tế lệch **2.18 điểm IoU** — sát ngoài biên. Không đủ
để dừng, nhưng phải nhớ khi đọc mọi kết quả bên dưới.

---

## Phần 2 — Kết quả

### Khoảng trống B — false negative / trần chất lượng bởi `M_prior`

**B1 — phân tầng theo kích thước GT** (999 ảnh, CI bootstrap 95%):

| nhóm | n | RPG-SAM | OP-SAM | PerSAM | `M_prior` |
|---|---|---|---|---|---|
| nhỏ (<5%) | 200 | 64.33 [59.77, 69.02] | 63.02 | 52.76 | 59.74 |
| vừa (5–15%) | 423 | 80.15 [78.01, 82.18] | 78.40 | 69.96 | 77.91 |
| lớn (>15%) | 376 | 78.80 [76.59, 80.89] | 77.11 | 65.73 | 74.64 |

Chênh lệch lớn − nhỏ của RPG-SAM: **14.46 điểm [9.40, 19.81]** → vượt ngưỡng 10 điểm.
Nhưng OP-SAM chênh 14.09 và PerSAM chênh 12.97 trên **cùng ảnh**. Suy giảm theo kích thước
là **đặc tính chung của bài toán one-shot**, không phải khuyết điểm riêng của RPG-SAM —
RPG-SAM thực ra nhỉnh hơn baseline ở cả ba nhóm.

**B2 — tỷ lệ trượt hoàn toàn** (IoU < 0.05) — đây mới là chỗ có phát hiện thật:

| phương pháp | toàn bộ | riêng polyp nhỏ |
|---|---|---|
| RPG-SAM | 3.00% [2.00, 4.00] | 12.50% [8.00, 17.50] |
| OP-SAM | 2.80% [1.80, 3.90] | 10.50% [6.50, 14.50] |
| PerSAM | 6.41% [4.90, 8.01] | 28.00% [22.00, 34.50] |

RPG-SAM hơn OP-SAM 1.63 điểm mIoU nhưng **không giảm được tỷ lệ trượt hoàn toàn** — thậm
chí nhích lên (12.50% so với 10.50% trên polyp nhỏ, CI chồng nhau nên chỉ kết luận được là
"không cải thiện"). Đúng tiêu chí B2 của kế hoạch.

→ **Gap B: XÁC NHẬN, nhưng phải phát biểu lại.** Không phải "RPG-SAM kém với polyp nhỏ hơn
baseline", mà là: **toàn bộ lợi thế mIoU của RPG-SAM đến từ việc tinh chỉnh những ảnh nó
vốn đã làm tương đối đúng; nó không đóng góp gì cho chế độ hỏng đáng lo nhất về lâm sàng là
bỏ sót polyp.** Và mIoU trung bình che mất điều đó — đúng như kế hoạch dự đoán.

### Khoảng trống C — giả định hình học của `S_geo`

**C1 — nguồn gốc `A_ref`** (đọc code): `A_ref` = diện tích polyp trong **ảnh support**,
tính động mỗi lần chạy ([`rpgsam/gas.py`](rpgsam/gas.py), `aref_mode="support"`). Với ảnh
support mặc định, `A_ref` = **17.11%** diện tích ảnh, trong khi GT trung vị chỉ **11.40%**.

**Hệ quả trực tiếp, đo trên chính ground truth:** `scale = min(1, A/A_ref)` **làm giảm điểm
67.6% số mask GT** (hệ số trung bình 0.649), trong đó 37.1% bị giảm xuống dưới 0.5. Nghĩa
là kể cả khi ứng viên **trùng khít GT**, Eq. 5 vẫn phạt nó chỉ vì nó nhỏ hơn polyp support.

**C2 — theo tứ phân vị solidity(GT)**:

| tứ phân vị | n | solidity | `S_geo`(GT) | scale(GT) | GT area% | RPG-SAM |
|---|---|---|---|---|---|---|
| Q1 (ít lồi nhất) | 250 | 0.892 | 0.730 | 0.818 | 22.3 | 70.38 |
| Q2 | 250 | 0.967 | 0.701 | 0.725 | 18.0 | 78.88 |
| Q3 | 249 | 0.990 | 0.636 | 0.643 | 13.6 | 80.89 |
| Q4 (lồi nhất) | 250 | 1.001 | 0.407 | 0.407 | 7.7 | 75.76 |

Q4 − Q1 = **+5.39 điểm [0.79, 9.98]** — có tương quan, nhưng **không đơn điệu** (Q3 mới cao
nhất), và `S_geo` đi **ngược chiều** với solidity vì Scale Consensus lấn át.

**Phase 4 — bảng chéo gỡ nhiễu kích thước × solidity:**

| ô | n | RPG-SAM |
|---|---|---|
| nhỏ × phẳng | 45 | 64.40 [54.39, 73.77] |
| nhỏ × lồi | 155 | 64.32 [58.75, 69.63] |
| lớn × phẳng | 455 | 75.64 [73.55, 77.73] |
| lớn × lồi | 344 | 84.63 [82.69, 86.45] |

Ở polyp **nhỏ**, solidity **không ảnh hưởng gì** (64.40 so với 64.32). Ở polyp **lớn**,
solidity đáng kể (9 điểm). Vậy hiệu ứng "phẳng" mà kế hoạch nghi ngờ chỉ tồn tại ở polyp
lớn, còn ở polyp nhỏ thì **kích thước mới là biến quyết định**.

**C3 — ablation trực tiếp công thức `S_geo`** (300 query, chênh lệch **ghép cặp theo từng
ảnh**, CI bootstrap 95%):

| biến thể | IoU | τ trung bình | Δ nhóm phẳng | Δ nhóm lồi | Δ toàn bộ |
|---|---|---|---|---|---|
| biến thể | IoU | prior | τ TB | Δ nhóm phẳng | Δ nhóm lồi | Δ toàn bộ |
|---|---|---|---|---|---|---|
| **`A_ref` = 5% ảnh** | **77.04** | 72.91 | 0.435 | | | |
| `A_ref` = 10% ảnh | 76.31 | 72.16 | 0.422 | | | |
| (iii) bỏ Scale Consensus | 76.24 | 71.94 | 0.459 | +0.19 [−2.04, 2.51] | +1.71 [−0.72, 4.38] | +0.91 [−0.79, 2.58] |
| (iv) perimeter convexity | 75.85 | 70.93 | 0.422 | +0.52 [−1.05, 2.21] | +0.53 [−0.23, 1.81] | +0.52 [−0.41, 1.53] |
| (iv)+(v) | 75.73 | 68.76 | 0.454 | +0.19 [−1.78, 2.33] | +0.63 [−0.35, 2.02] | +0.40 [−0.77, 1.67] |
| **gốc (Eq. 5)** | **75.32** | 71.41 | 0.410 | — | — | — |
| (ii) bỏ Weighted Solidity | 75.15 | 71.27 | 0.400 | +0.13 [−0.24, 0.50] | −0.50 [−1.20, −0.01] | −0.17 [−0.54, 0.15] |
| ngưỡng cố định τ = 0.40 | 75.15 | 71.27 | 0.400 | — | — | — |
| (v) phạt scale hai phía | 75.10 | 67.84 | 0.463 | −0.85 [−3.23, 1.38] | +0.48 [−0.57, 1.94] | −0.22 [−1.57, 1.20] |
| (v) + `A_ref` = 10% | 73.96 | 62.41 | 0.529 | | | |
| ngưỡng cố định τ = 0.70 | 65.45 | 43.63 | 0.700 | — | — | — |

**Tiêu chí C3 của kế hoạch KHÔNG đạt.** Kế hoạch đòi biến thể (ii)/(iv) *"cải thiện rõ trên
nhóm phẳng"*; thực tế **mọi CI ghép cặp đều chứa 0**. Giả thuyết "`S_geo` thiên kiến chống
polyp phẳng, gỡ thiên kiến ra thì nhóm phẳng khá lên" **không đứng vững** ở cỡ mẫu này.

Nhưng một kết quả khác, sạch hơn, xuất hiện: **bỏ Weighted Solidity cho ra 75.15 / prior
71.27 — trùng khít từng chữ số với ngưỡng cố định τ = 0.40.** Đó là chứng minh bằng số của
sự thoái hoá: khi wsol = 1, `S_geo = min(1, A/A_ref)` mà diện tích giảm đơn điệu theo τ,
nên GAS **luôn** chọn đáy dải quét — nó *đúng bằng định nghĩa* là một ngưỡng cố định. Toàn
bộ số hạng solidity, thứ duy nhất ngăn sự thoái hoá hoàn toàn, chỉ đáng **0.17 điểm IoU**.

**Giả thuyết sửa chữa của chính tôi cũng bị bác bỏ.** Tôi dự đoán rằng trả lại vế phạt còn
thiếu (`min(A/A_ref, A_ref/A)`) sẽ chữa được sự thoái hoá. Thực nghiệm nói ngược: phạt hai
phía làm **tệ đi ở mọi cấu hình** (prior 67.84 / 62.41 / 68.76 so với 71.41 của bản gốc).
Lý do: nó ép diện tích mask về đúng `A_ref`, trong khi polyp thật trải từ 0.47% đến 81.18%
diện tích ảnh — **không tồn tại một thang tham chiếu duy nhất nào đúng**. Tính một phía của
Eq. 5 là nguyên nhân gây thoái hoá, nhưng thêm vế còn lại không phải là cách chữa.

**Điều thực sự quan trọng lại là `A_ref`, và ở đây bài tự mâu thuẫn.** Bài nói Scale
Consensus tồn tại để *"prevent small, fragmented noise from obtaining artificially high
solidity scores"*, đồng thời nói `A_ref` *"represents the expected median scale of polyps"*.
**Hai câu này không tương thích nhau.** Nếu `A_ref` đúng bằng thang trung vị của polyp thì
số hạng đó phạt luôn phần lớn polyp hợp lệ — đo được: **67.6% mask GT bị giảm điểm**, hệ số
trung bình 0.649. Muốn làm đúng việc mà văn bản mô tả (chỉ dập nhiễu vụn), `A_ref` phải
**thấp hơn hẳn** trung vị. Và thực nghiệm xác nhận đúng như vậy: `A_ref` = 5% diện tích ảnh
(so với trung vị GT 11.4%) cho kết quả tốt nhất, **+1.72 IoU** so với cách đọc "diện tích
polyp support".

→ **Gap C: XÁC NHẬN, nhưng cả cơ chế lẫn tiêu chí của kế hoạch đều sai.** Không đo được
hiệu ứng "phạt polyp phẳng". Ba điều đo được là: (1) `S_geo` chỉ đáng **0.17 điểm** so với
ngưỡng cố định tốt nhất, và bỏ solidity ra thì nó **trùng khít** ngưỡng cố định τ = 0.40;
(2) tham số **quan trọng nhất là `A_ref`**, thứ mà bài định nghĩa mâu thuẫn và không nói
cách tính trong bối cảnh one-shot; (3) chọn `A_ref` tuỳ tiện bằng 5% ảnh **vượt công thức
gốc 1.72 điểm**. Nhận định "thích ứng theo hình học" không có nội dung đo được ở đây — cái
thực sự chi phối kết quả là một hằng số mà bài để ngỏ.

⚠️ **Giới hạn thống kê:** ở n = 300 các CI quá rộng để phân định giữa các biến thể (ước
lượng điểm thiên về (iii) +0.91 và (iv) +0.52, nhưng cả hai đều chứa 0). Cần chạy bản đầy
đủ 999 query trên Kaggle (`--c3-limit 0`) để thu hẹp CI khoảng 1.8 lần trước khi kết luận
biến thể nào thực sự tốt hơn.

### Khoảng trống A — lan truyền lỗi qua pipeline

**A2 — tương quan `M_prior` ↔ kết quả cuối:** Pearson r = **0.923** [0.903, 0.940],
Spearman ρ = 0.807 — vượt xa ngưỡng 0.7. Nhưng như mục 1.3 đã nói, con số này gần như tất
yếu. Phần bác bỏ được là **recovery theo dải chất lượng prior**:

| dải prior IoU | n | prior | final | recovery |
|---|---|---|---|---|
| [0.0, 0.1) | 33 | 3.39 | 2.29 | **−1.18% [−2.08, −0.44]** |
| [0.1, 0.3) | 57 | 20.36 | 19.19 | **−1.25% [−4.14, 1.68]** |
| [0.3, 0.5) | 53 | 39.98 | 48.04 | 14.16% [5.71, 23.25] |
| [0.5, 0.7) | 136 | 60.85 | 67.36 | 16.77% [10.90, 22.61] |
| [0.7, 0.9) | 528 | 82.64 | 86.60 | 23.28% [19.68, 26.69] |
| [0.9, 1.0) | 192 | 92.02 | 92.68 | 7.25% [1.05, 13.22] |

Recovery **âm** khi prior < 0.3 và **cao nhất** khi prior đã ở 0.7–0.9. Nếu PIR thật sự sửa
được lỗi thượng nguồn thì thứ tự phải ngược lại. **PIR là bộ tinh chỉnh, không phải bộ sửa
lỗi** — đúng như kế hoạch nghi ngờ.

**A3 — định lượng chế độ hỏng:** trong 30 ảnh RPG-SAM trượt hoàn toàn, **21 ảnh (70.0%) đã
có `M_prior` rỗng ngay từ đầu**; GT trung vị của nhóm này chỉ chiếm 2.26% diện tích ảnh.
Trên toàn bộ 999 query, vòng PIR **cải thiện 63.7%** số ảnh (trung bình +0.080 IoU) nhưng
**làm xấu đi 21.8%** (trung bình −0.077); có 8 ảnh mà prior trên 0.5 IoU bị chính các
negative prompt của vòng lặp kéo tụt hơn 0.25.

**Panel định tính** ([`outputs/figures/`](outputs/figures/)) cho thấy hai chế độ hỏng tách
bạch, và chế độ thứ hai mới là chế độ chỉ ra bản chất vấn đề:

*`a3_upstream_miss.png` — lỗi từ thượng nguồn.* Cả 4 ca, prompt của PIR đều rơi **vào bên
trong** vùng `M_prior` vốn đã sai, và mask cuối chỉ là bản làm sạch của vùng sai đó. PIR
không bao giờ tìm ra ngoài prior. Đúng như kế hoạch A3 dự đoán: *"PIR chỉ chỉnh biên, không
mở rộng/sửa vùng bị bỏ sót hoàn toàn"*. Đáng chú ý ở một ca, heatmap **rất tự tin và khu trú
tốt** — nhưng vào một khối polyp-like khác, không phải GT được chú thích. Đó là lỗi khớp
ngữ nghĩa, không phép thích ứng ngưỡng nào cứu được.

*`a3_loop_ruined_prior.png` — vòng lặp phá hỏng prior tốt.* Cả 4 ca, `M_prior` **khoanh đúng
polyp** (một ca tìm đúng cả hai polyp), rồi PIR thêm 3–4 prompt âm và mask cuối **sụp gần
như trắng**: IoU rơi từ ~0.75 xuống 0.08 / 0.22 / 0.32 / 0.27.

Panel thứ hai chỉ ra cơ chế chính xác hơn cả giả thuyết ban đầu: **hàm mục tiêu của PIR là
sự đồng thuận với `M_prior`, không phải sự đúng đắn.** Vòng lặp đo Cov và IoU *so với
prior*, và mask được chọn cuối cùng là mask có IoU **cao nhất so với prior**. Hệ quả kép:
(1) chất lượng pipeline bị **chặn trên bởi `M_prior` theo đúng thiết kế**, và (2) khi SAM2 —
vốn có tiên nghiệm vật thể riêng — đoán đúng hơn `M_prior`, PIR **chủ động phá bỏ** lợi thế
đó để kéo mask về khớp với prior.

**A1 — làm hỏng RWPM có kiểm soát, giữ nguyên GAS+PIR** (300 query). `recovery` =
`(IoU_final − IoU_prior) / (1 − IoU_prior)` tính theo từng ảnh — phần headroom còn lại sau
RWPM mà SAM2+PIR thực sự khép được:

| biến thể RWPM | prior IoU | final IoU | Δ | recovery |
|---|---|---|---|---|
| baseline (RWPM đầy đủ) | 71.41 | 75.32 | +3.91 | 19.32% [14.85, 23.48] |
| (i) bỏ background suppression | 68.56 | 73.33 | +4.77 | 22.69% [19.06, 26.15] |
| (ii) `W_k` đều | 69.31 | 72.91 | +3.60 | 17.66% [13.68, 21.59] |
| (i)+(ii) | **63.66** | 67.09 | +3.43 | **17.13%** [13.62, 20.75] |
| (iii) support có vệt phản chiếu | 70.19 | 74.39 | +4.20 | 20.87% [16.67, 24.96] |
| (iii) support bị làm mờ | 69.66 | **76.49** | +6.83 | 25.35% [21.15, 29.44] |
| (iii) mask support bị co | 65.74 | 69.12 | +3.38 | **13.52%** [9.16, 17.64] |
| (iii) mask support bị giãn | 69.53 | 73.48 | +3.95 | 19.39% [15.12, 23.50] |
| (iii) mask support lệch chỗ | 64.61 | 70.03 | +5.42 | 19.29% [15.34, 23.15] |

**Tiêu chí A1 ĐẠT:** mọi biến thể suy giảm đều có recovery **dưới 30%** (13.52–25.35%).

Nhưng bằng chứng mạnh hơn nằm ở **hình dạng** của cột recovery: nó **gần như phẳng** dù
prior trải 7.75 điểm (71.41 → 63.66). Nếu PIR sửa được lỗi thượng nguồn thì recovery phải
**tăng** khi prior tệ đi — còn nhiều headroom hơn để sửa. Thực tế biến thể tệ nhất
(i)+(ii), prior 63.66, lại có recovery **17.13%, thấp hơn** baseline. Trường hợp mask
support bị co cho thấy rõ nhất cơ chế: prior teo lại một cách hệ thống, và PIR — vốn chỉ có
điểm dương trên vùng false-negative *của prior* — không có cách nào biết là phải nở ra,
nên recovery rơi xuống 13.52%.

*(Một điểm lạ đáng ghi nhận: làm mờ ảnh support cho final IoU **cao nhất bảng** 76.49, hơn
cả baseline, dù prior kém hơn. Làm mờ có vẻ hoạt động như một phép chính quy hoá lên
prototype. Đây là kết quả ngoài dự kiến, chưa giải thích được, cần lặp lại ở 999 query.)*

→ **Gap A: XÁC NHẬN.** PIR là bộ tinh chỉnh biên, không phải bộ sửa lỗi. Recovery không đổi
theo chất lượng prior, recovery **âm** khi prior < 0.3, 70% ca trượt hoàn toàn do prior
rỗng, và vòng lặp còn làm xấu đi 21.8% số ảnh.

---

## Phần 3 — Tổng hợp

| Khoảng trống | Kế hoạch dự đoán | Kết quả | Ghi chú |
|---|---|---|---|
| **A** | PIR không sửa được lỗi thượng nguồn | ✅ **xác nhận** (A1 đạt tiêu chí, A2 + A3 củng cố) | recovery 13.5–25.4% ở mọi biến thể suy giảm và **không tăng** khi prior tệ đi; recovery âm ở prior < 0.3; 70% ca trượt hoàn toàn do prior rỗng; PIR làm xấu 21.8% ảnh. Cơ chế: **hàm mục tiêu của PIR là khớp `M_prior`, không phải đúng** |
| **B** | Bỏ sót polyp nhỏ, mIoU che mất | ✅ xác nhận, **phát biểu lại** | suy giảm theo kích thước là chung cho mọi baseline; điều đặc hiệu là RPG-SAM **không giảm** tỷ lệ trượt hoàn toàn dù mIoU cao hơn |
| **C** | `S_geo` phạt polyp phẳng | ⚠️ **tiêu chí kế hoạch không đạt**, gap xác nhận theo hướng khác hẳn | không đo được hiệu ứng "phạt polyp phẳng" (mọi CI chứa 0); `S_geo` chỉ đáng **+0.17 IoU**, bỏ solidity ra thì **trùng khít** τ=0.40 cố định; biến quyết định là **`A_ref`** — bài định nghĩa mâu thuẫn, và đặt `A_ref` = 5% ảnh **hơn công thức gốc 1.72 điểm** |

**Ba khoảng trống không độc lập.** Chúng gặp nhau ở đúng một chỗ: polyp nhỏ hơn polyp
support bị Scale Consensus phạt (C) → `M_prior` yếu hoặc rỗng (B) → PIR không cứu được, có
khi còn làm tệ hơn (A). Ô "nhỏ × phẳng" (n = 45, IoU 64.40) và 21 ca prior rỗng là nơi cả
ba cùng biểu hiện.

**Và có một nguyên nhân chung nằm dưới cả ba.** Cả GAS lẫn PIR đều lấy `M_prior` làm chuẩn
mực: GAS chọn ngưỡng theo hình học của ứng viên chứ không theo bằng chứng ảnh, PIR tối ưu
sự khớp với chính prior mà GAS vừa tạo ra. Không có bước nào trong pipeline có thể phát
hiện rằng `M_prior` đã sai ngay từ đầu. Đó là lý do khoảng cách giữa `M_prior` và kết quả
cuối gần như cố định (+3.4 đến +4.8 điểm) bất kể prior tốt hay xấu — **RPG-SAM có trần
chất lượng bằng đúng chất lượng của RWPM, theo thiết kế chứ không phải do lỗi triển khai.**

### Còn thiếu gì

| | |
|---|---|
| **C4** (10 support draws) và **B3** (3 run) | chưa chạy — để dành Kaggle, ~55 phút |
| **C3 ở 999 query** | bản 300 query có CI quá rộng để phân định các biến thể `S_geo` |
| 5 cột còn lại của Table 1 | CVC-ClinicDB / ColonDB / PolypGen / Piccolo đều sau đăng ký |
| ProtoSAM, Matcher, SEGIC | chưa cài lại nên nhận định "vượt ProtoSAM 5.56 điểm" không kiểm chứng được |

**Điều công bằng phải nói với tác giả.** Sáu quyết định triển khai phải tự suy ra — nhất là
chuẩn hoá của Eq. 4 (hai cách đọc lệch nhau 28 điểm) và định nghĩa `A_ref` — đủ sức làm
dịch chuyển mọi baseline. Nếu heatmap của tác giả có ngưỡng tối ưu quanh 0.7 thay vì 0.4
như ở đây, phần lớn phân tích về GAS sẽ phải làm lại. **Ngoại lệ duy nhất là tính một phía
của Eq. 5** — đó là tính chất của công thức, không phải của bản triển khai.
