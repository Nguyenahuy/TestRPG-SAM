# RPG-SAM — kiểm chứng ba khoảng trống nghiên cứu

Bộ thí nghiệm kiểm chứng ba nhận định về **RPG-SAM** ("Reliability-Weighted Prototypes and
Geometric Adaptive Threshold Selection for Training-Free One-Shot Polyp Segmentation",
[arXiv:2603.07436](https://arxiv.org/abs/2603.07436)):

| | Khoảng trống | Câu hỏi |
|---|---|---|
| **A** | Lan truyền lỗi qua pipeline tuần tự | `H_init`/`M_prior` sai thì PIR có sửa được không, hay chỉ tinh chỉnh biên trên nền đã sai? |
| **B** | Thiếu phân tích false negative, chất lượng bị trần bởi `M_prior` | RPG-SAM có bỏ sót polyp nhỏ không? |
| **C** | Giả định hình học của `S_geo` không tổng quát | `S_geo` có hệ thống đánh giá thấp polyp phẳng / lệch xa `A_ref` không? |

Vì tác giả chưa phát hành code ("Code will be released"), đây là một **replication** dựng
lại từ công thức (1)–(5) và Section 2.1–2.3, không phải reproduction. Baseline replication
đạt **76.47 mIoU / 83.36 mDice** trên Kvasir-SEG so với 78.65 / 85.65 công bố, và AUC-PR
của heatmap lệch 0.002 — đủ gần để các thí nghiệm bên dưới có ý nghĩa.

> ⚠️ Mọi kết luận ở đây nói về **bản replication này**, không phải về code gốc của tác giả.
> Sáu quyết định triển khai phải tự suy ra (đặc biệt là chuẩn hoá của Eq. 4, và định nghĩa
> `A_ref`) đủ sức làm dịch chuyển baseline của bảng ablation. Ngoại lệ duy nhất là các phê
> phán rút ra từ **bản thân công thức** — xem phần `S_geo` bên dưới.

---

## Chạy cái gì, ở đâu

| Phase | Thí nghiệm | Cần GPU? | Thời gian | Trạng thái |
|---|---|---|---|---|
| 0 | Baseline replication (Table 1/2) | có | ~17 ph | ✅ có sẵn trong `outputs/tables/` |
| 1–2 | B1, B2, C2, A2, A3, Phase 4 | **không** | ~1 ph | ✅ chạy lại được ngay, kết quả ở `outputs/analysis/` |
| 3 | **C3** — ablation `S_geo` (11 run) | có | ~1.0 h | ✅ **đã chạy**, JSON kèm repo |
| 3 | **A1** — fault injection (13 run) | có | ~0.9 h | ✅ **đã chạy**, JSON kèm repo |
| 3 | **A3** — panel định tính | có | ~1 ph | ✅ **đã chạy**, PNG ở `outputs/figures/` |
| 3 | **C4** — độ nhạy support / `A_ref` (10 run) | có | ~50 ph | ⬜ **để dành Kaggle** |
| 3 | **B3** — subset kích thước cực đoan (3 run) | có | ~5 ph | ⬜ **để dành Kaggle** |
| 3 | C3 ở 999 query (tuỳ chọn) | có | ~2.5 h | ⬜ để thu hẹp CI |

Kết quả của 24 run đã hoàn thành **được commit thẳng vào `outputs/gaps/`**. Driver
`run_gaps.py` bỏ qua mọi tag đã có JSON, nên trên Kaggle nó chỉ chạy **13 run còn thiếu**.

Phase 1–2 chạy được mà **không cần GPU**, vì kết quả per-image của cả RPG-SAM, OP-SAM và
PerSAM đã commit sẵn trong `outputs/tables/`. Chỉ `gt_stats.py` cần bộ mask Kvasir-SEG.

---

## Chạy trên máy local (Phase 1–2, ~1 phút)

```bash
pip install -r requirements.txt

export RPGSAM_DATA=/duong/dan/toi/thu-muc-chua/Kvasir-SEG   # chỉ cần cho gt_stats.py
python scripts/gt_stats.py          # hình học GT cho từng ảnh -> outputs/analysis/gt_stats.json
python scripts/analyze_gaps.py      # B1, B2, C2, A2, Phase 4 -> outputs/analysis/GAPS.md
```

Không cần DINOv2, không cần SAM2, không cần checkpoint.

---

## Chạy trên Kaggle (Phase 3)

### 1. Đưa repo lên GitHub

```bash
cd rpg-sam-gaps
git init && git add -A
git commit -m "RPG-SAM gap study"
git remote add origin https://github.com/<TEN-CUA-BAN>/rpg-sam-gaps.git
git push -u origin main
```

`data/`, `checkpoints/` và `outputs/gaps/` đã nằm trong `.gitignore` nên repo chỉ vài trăm KB.

### 2. Chuẩn bị notebook Kaggle

1. Vào [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. **File → Import Notebook** → upload `notebooks/kaggle_rpgsam_gaps.ipynb`.
   (Nếu bạn sửa `.py` thì chạy lại `python scripts/make_notebook.py` để sinh lại `.ipynb`.)
3. Panel bên phải, mục **Session options**:
   - **Accelerator**: `GPU T4 x2` hoặc `GPU P100`
   - **Internet**: **On** — bắt buộc, vì DINOv2 tải qua `torch.hub` và SAM2 cài từ GitHub.
   - **Persistence**: `Files only` nếu bạn định chạy nhiều session nối tiếp nhau.
4. **Add Input → Datasets** → tìm `kvasir-seg` (ví dụ dataset công khai
   `debeshjha1/kvasirseg`) và thêm vào. Cell 3 tự dò thư mục nào có `images/` + `masks/`.
5. Sửa biến `REPO` ở cell đầu thành URL fork của bạn.

### 3. Chạy

Chạy tuần tự từ cell 1:

| cell | việc | thời gian |
|---|---|---|
| 1–3 | clone repo, cài SAM2, tải checkpoint | ~3 ph |
| 4 | trỏ đường dẫn dataset | tức thì |
| 5 | **smoke test 8 ảnh** — nếu qua thì DINOv2 + SAM2 + data đều ổn | ~1 ph |
| 6 | Phase 1–2 (CPU) | ~1 ph |
| 7 | `--dry-run`: xem còn đúng 13 run cần chạy | tức thì |
| 8 | **C4** (10 run) | ~50 ph |
| 9 | **B3** (3 run) | ~5 ph |
| 10–11 | sinh báo cáo + zip kết quả | ~1 ph |

**Đừng bỏ qua cell 5.** Nó tốn 1 phút và bắt được mọi lỗi môi trường trước khi bạn tiêu
gần một giờ GPU.

**Quota Kaggle:** 30 giờ GPU/tuần, tối đa 9 giờ mỗi session. Phần còn lại chỉ ~1 giờ nên
thừa sức một session. `run_gaps.py` **resumable** — tag nào đã có JSON thì bỏ qua — nên nếu
session hết giờ, chỉ cần chạy lại đúng cell đó, miễn là bật **Persistence: Files only** để
`outputs/gaps/` sống sót qua các session.

**Tuỳ chọn — C3 ở 999 query (~2.5 h).** Bản C3 đã chạy dùng 300 query và **mọi khoảng tin
cậy ghép cặp đều chứa 0**, tức chưa phân định được các biến thể `S_geo`. Chạy đủ 999 query
sẽ thu hẹp CI khoảng 1.8 lần. Cần `--force` vì JSON 300-query đã nằm sẵn trên đĩa:

```python
subprocess.run([sys.executable, "scripts/run_gaps.py",
                "--group", "C3", "--c3-limit", "0", "--force"])
```

### 4. Lấy kết quả về

Cell cuối zip `outputs/` thành `/kaggle/working/rpgsam_gap_outputs.zip`. Bấm **Save
Version → Save & Run All (Commit)**, xong vào tab **Output** của version để tải về.

---

## Cấu trúc

```
rpgsam/                 RPG-SAM dựng lại từ bài báo
  rwpm.py               Sec. 2.1, Eq. 1-4  prototype + trọng số tin cậy W_k = C_k·R_k
  gas.py                Sec. 2.2, Eq. 5    quét ngưỡng + S_geo  (+ 5 biến thể cho C3)
  pir.py                Sec. 2.3           vòng lặp tinh chỉnh SAM2
  pipeline.py           RWPM -> GAS -> PIR
  faults.py             các phép làm hỏng support có kiểm soát (A1)
opsam/                  backbone đông cứng, dataset adapter, metric — dùng chung với
                        replication OP-SAM để hai bài được chấm bằng đúng một bộ code
scripts/
  run_eval.py           một cấu hình = một lần chạy = một file JSON
  run_gaps.py           driver cho toàn bộ lưới Phase 3, resumable
  gt_stats.py           hình học ground-truth từng ảnh (CPU)
  analyze_gaps.py       B1, B2, C2, A2, Phase 4  (CPU, hậu kỳ)
  report_gaps.py        A1, C3, C4, B3 -> GAP_REPORT.md
  visualize.py          panel định tính cho A3
outputs/
  tables/               kết quả per-image đã commit (baseline + OP-SAM + PerSAM)
  analysis/             GAPS.md, GAP_REPORT.md sinh ra từ các script trên
```

## Thiết kế thí nghiệm

Mỗi biến thể là một **flag CLI**, không phải một nhánh code, nên mọi kết quả tái lập được
bằng đúng dòng lệnh đã ghi trong `cfg` của file JSON tương ứng.

**C3 — năm cách đọc lại Eq. 5** (`--geo-mode`). Công thức gốc là
`S_geo = weighted_solidity × min(1, A/A_ref)`:

| mode | thay đổi |
|---|---|
| `full` | Eq. 5 nguyên bản |
| `no_solidity` | bỏ Weighted Solidity |
| `no_scale` | bỏ Scale Consensus |
| `perim` | thay solidity diện tích bằng **perimeter convexity** `P(hull)/P(contour)` — polyp sessile lõm nhưng *lõm trơn*, chỉ số này không phạt nó nặng như solidity |
| `sym_scale` | `min(A/A_ref, A_ref/A)` — trả lại vế phạt còn thiếu |
| `perim_sym` | cả hai sửa chữa |

`sym_scale` nhắm thẳng vào một phê phán **không phụ thuộc cách triển khai**: Eq. 5 như đã
viết chỉ phạt mask *nhỏ hơn* `A_ref`, hoàn toàn không phạt mask *lớn hơn*. Vì diện tích
vùng kích hoạt giảm đơn điệu khi tăng ngưỡng, `S_geo` gần như đơn điệu giảm theo `τ`, và
GAS thoái hoá thành "chọn đáy dải quét" — 90.8% số ảnh trong bản replication này.

**A1 — làm hỏng RWPM, giữ nguyên GAS+PIR** (`--no-bg-suppress`, `--no-reliability`,
`--fault`). Các phép làm hỏng support: `specular` (dán vệt phản chiếu — đúng thứ mà
Contrast Factor tuyên bố chống lại), `blur`, `noise`, `erode`/`dilate` (nhiễu chú thích),
`shift` (mask lệch chỗ). Chỉ số kết luận là **recovery** = phần headroom còn lại sau RWPM
mà SAM2+PIR thực sự khép được, tính theo từng ảnh nên có khoảng tin cậy.

**C4 — support chọn theo phân vị diện tích polyp**, không phải ngẫu nhiên. Vì `A_ref`
*chính là* diện tích polyp support, sweep này đồng thời là phép quét `A_ref` từ 0.47% đến
81.18% diện tích ảnh.

## Ghi chú thống kê

Các nhóm phân tầng nhỏ (n = 45 cho ô "nhỏ × phẳng"), nên mọi thống kê nhóm đều kèm
**khoảng tin cậy bootstrap 95%**, và so sánh giữa hai lần chạy được **ghép cặp theo từng
ảnh query** vì hai lần chạy dùng chung tập query.

## Dữ liệu

Chỉ **Kvasir-SEG** khả dụng. CVC-ClinicDB, CVC-ColonDB, ba trung tâm PolypGen và Piccolo
đều sau đăng ký, nên năm cột còn lại của Table 1 không kiểm chứng được. Kvasir mang nhận
định chủ đạo của bài (78.65 mIoU) nên đó là thứ được kiểm ở đây. `load_dataset` đã nhận
layout `images/` + `masks/`, thả thư mục mới vào `data/` là chạy được.
