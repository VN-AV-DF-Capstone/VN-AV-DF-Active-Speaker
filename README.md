# VN-AV-DF-Active-Speaker
---

## 1. Giới Thiệu (Context & Motivation)

Trong pipeline tiền xử lý dữ liệu phát hiện giả mạo âm thanh - hình ảnh tiếng Việt (**VN-AV-DF-Capstone**), module **Active Speaker Detection (ASD) & ROI Tracking** đóng vai trò là **Chốt chặn chất lượng (Critical Quality Gate)**:
1. **Loại bỏ dữ liệu bẩn**: Tự động phát hiện và loại bỏ các đoạn video thuyết minh/lồng tiếng (**Voiceover**), cảnh quay chèn tư liệu (**B-roll**), hoặc khuôn mặt tĩnh/ảnh ghép đơ miệng (**Mouth Freeze / Static Face**).
2. **Giải quyết tranh chấp đa khuôn mặt (Multi-Speaker Competition)**: Trong video talkshow/podcast có từ 2–3 người cùng khung hình, hệ thống phải chỉ định chính xác ai là người đang phát ra giọng nói, ai chỉ là người ngồi nghe (người nghe cười, gật đầu, mấp máy môi không được gán nhầm là speaker).
3. **Bàn giao vùng quan tâm (Active Face ROI Handover)**: Cung cấp tọa độ Bounding Box được làm mượt theo thời gian của người nói thực sự, phục vụ trực tiếp cho các mô hình học sâu downstream (như **AV-HuBERT**, **LipForensics**, **SyncNet**).

---

## 2. Kiến Trúc 3-Tier Cascade (Phân Tầng Ngắn Mạch Tiết Kiệm Tải)

Hệ thống được thiết kế theo cơ chế **3-Tier Cascade** phân tầng thông minh, giúp giải quyết $\ge 80\%$ video ở các tầng siêu nhẹ mà không làm nghẽn GPU:

```
                            [ Raw Video Clip (25 fps, 16kHz Mono) ]
                                                │
════════════════════════════════════════════════╪════════════════════════════════════════════════
[ TIER 0: Ultra-Light Fast Gating ]             │
(CPU / ONNX Runtime)                            ├──────────────────────────┐
  • Silero VAD v5 (ONNX, 200ms bins)            ▼                          ▼
  • Light Face Detection (InsightFace / OpenCV)[Speech = False]       [Face Count = 0]
                                                │                          │
                                                ▼                          ▼
                                       Label: "silent"            Label: "voiceover"
                                      (Skip Deep Models)          (Auto Reject / Short-circuit)
════════════════════════════════════════════════╪════════════════════════════════════════════════
[ TIER 1: High-Throughput Screening ]           │ (Có tiếng & có ít nhất 1 mặt)
(GPU / Fast PyTorch / 1.1M params)              ▼
  • Face Tracking & Linear Interpolation (Nội suy 50% frame)
  • Mouth Articulation Flow Metric (AbsDiff vi sai mức xám Δ)
  • Light-ASD Model Inference (13-dim MFCC + Bi-GRU)
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
           [ Confident Bins ]                                     [ Ambiguous Bins ]
   (1 Mặt rõ, điểm ASD cao/thấp dứt khoát,                 (Điểm ASD lấp lửng [ -margin, +margin ],
    Cơ miệng dao động nhịp nhàng)                          Nhiều mặt cạnh tranh điểm sát nhau,
                     │                                     hoặc Điểm cao nhưng môi bị đơ "mouth_frozen")
                     │                                                     │
═════════════════════╪═════════════════════════════════════════════════════╪════════════════════
[ TIER 2: Deep Arbiter & Lip-Sync Verifier ]                               │ (Chỉ chạy ~15-20% bins)
(LoCoNet + LASER WACV'26)                                                  ▼
                                                          • Inter-Speaker SIM (Gỡ rối đa người)
                                                          • Lip-Centric Attention (So khớp âm-môi)
                                                          • Sinh điểm LASER Score [0.0 .. 1.0]
═══════════════════════════════════════════════════════════════════════════╪════════════════════
[ DECISION & ROI HANDOVER ]                                                │
                                                                           ▼
                                                    • Pure Temporal Policy Engine (policy.py)
                                                    • Temporal Decision: PASS / REJECT / MANUAL
                                                    • Smoothed Active Face ROI Trajectory (BBox/Crop)
```

---

## 3. Cấu Trúc Thư Mục & Vai Trò Các Thành Phần

```
VN-AV-DF-Active-Speaker/
├── configs/
│   └── default_poc.yaml             # Cấu hình ngưỡng thời gian 200ms, margin, tham số VAD & ASD
├── data/
│   └── poc_manifest.csv             # Tập 64 clips trích xuất từ nhãn review thực tế (Clean, Mouth defect, Voiceover, Multi-face)
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_types.py            # Dataclasses: BoundingBox, FaceTrack, BinEvidence, ClipResult
│   │   └── policy.py                # Pure temporal policy (200ms binning, fail-closed, zero-torch)
│   ├── frontends/
│   │   ├── __init__.py
│   │   ├── vad.py                   # Silero VAD (pure ONNX Runtime CPU, không torchaudio)
│   │   ├── face_tracker.py          # Multi-face tracker (nội suy 50% frame, hỗ trợ InsightFace/MediaPipe/OpenCV)
│   │   └── mouth_motion.py          # Vi sai quang sai vùng miệng (absdiff flow metric để bắt mouth freeze)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── light_asd.py             # Model Light-ASD standalone (~1.1M params, Separable Conv + Bi-GRU)
│   │   └── laser_adapter.py         # Adapter Tier 2 Arbiter (LoCoNet + LASER) cho ambiguous bins
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── cascaded_engine.py       # Bộ điều phối trung tâm 3-Tier Cascade
│   │   └── roi_extractor.py         # Trích xuất và làm mượt quỹ đạo ROI (Smoothed Active BBox/Crop)
│   └── utils/
│       ├── __init__.py
│       ├── audio_video.py           # Decode video 25fps và audio 16kHz mono (PyAV, không phụ thuộc ffmpeg CLI)
│       ├── dashboard.py             # Bộ vẽ Diagnostic Dashboard 5 tầng (Mel-Spec, F0 Pitch, Sync Heatmap)
│       └── visualizer.py            # Render video MP4 trực quan (BBox xanh/xám, thanh HUD thông số)
├── tools/
│   ├── download_weights.py          # Script tải tự động trọng số Silero VAD và Light-ASD
│   ├── prepare_poc_dataset.py       # Script quét tự động dữ liệu review và tạo poc_manifest.csv
│   └── evaluate_poc.py              # Script đo đạc tự động Precision, Recall, Rejection Rate, FPS
├── run_poc.py                       # CLI chính để chạy thử nghiệm 1 video hoặc batch
├── requirements.txt                 # Danh sách thư viện phụ thuộc tối thiểu
└── README.md                        # Tài liệu hướng dẫn sử dụng module
```

---

## 4. Hướng Dẫn Thao Tác Trực Tiếp trên Máy LOCAL (Windows)

### Bước 1: Mở Terminal và chuyển vào thư mục module
```powershell
cd "L:\FPT\Side project\Capstone\resources\VN-AV-DF-Active-Speaker"
```

### Bước 2: Cài đặt thư viện phụ thuộc
*(Không yêu cầu cài đặt phần mềm ngoài `ffmpeg.exe` vì module sử dụng PyAV `av` giải mã trực tiếp trong Python)*
```powershell
pip install -r requirements.txt
```

### Bước 3: Tải tự động trọng số mô hình nhẹ
Tải Silero VAD ONNX (~2MB) và Light-ASD checkpoint (~4.5MB) vào thư mục `weights/`:
```powershell
python tools/download_weights.py
```

### Bước 4: Tạo tập dữ liệu POC thử nghiệm (Nếu cần tái tạo lại)
Script tự động quét file review của nhóm và tạo ra tập 64 clips đại diện:
```powershell
python tools/prepare_poc_dataset.py
```

### Bước 5: Chạy thử nghiệm trên 1 Clip đơn lẻ (Xuất Video HUD + Biểu đồ chẩn đoán)
```powershell
python run_poc.py --video "L:\FPT\Side project\Capstone\resources\VN-AV-DF-Capstone\data\manifests\dataset_v1\reviews\exports\clips\nguyenlamanh\5CNS_gaH16c_s0000653218_e0000658270.mp4" --visualize
```

* **Kết quả nhận được tại thư mục `results/<clip_id>/`**:
  * `result.json`: Chi tiết quyết định (`pass`, `reject`, `manual`), lý do, tỷ lệ thời gian nói, tier thoát sớm.
  * `active_roi.json`: Tọa độ Bounding Box từng frame kèm nhãn `is_speaking`.
  * `..._annotated.mp4`: Video trực quan hóa vẽ khung màu (Xanh = Người nói chính, Xám = Người nghe) và thanh HUD thông số thời gian thực.
  * `..._diagnostic_dashboard.png`: **Biểu đồ đa tầng độ phân giải cao** gồm:
    1. *Log Mel-Spectrogram* + *Đường cong cao độ $F_0$ (Pitch Hz)*.
    2. *Acoustic RMS Energy* đối chiếu trực tiếp với *Lip Articulation Dynamics*.
    3. *Cross-Modal Synchronization Heatmap* (bản đồ nhiệt tương quan âm-hình theo độ trễ thời gian).
    4. *Multi-Tier Scoring Timeline* (điểm Light-ASD và LASER).
    5. *200ms Decision Ribbon* (dải màu phân loại trạng thái từng bin).

### Bước 6: Chạy đánh giá Benchmark toàn bộ tập POC (Batch Evaluation)
```powershell
python tools/evaluate_poc.py --manifest data/poc_manifest.csv --out_dir results/evaluation
```

---

## 5. Hướng Dẫn Thao Tác trên KAGGLE NOTEBOOK (GPU T4 Miễn Phí)

Nếu nhóm muốn tận dụng GPU Tesla T4 của Kaggle để kiểm thử tốc độ xử lý hàng loạt:

### Bước 1: Khởi tạo Kaggle Notebook
1. Đăng nhập [Kaggle](https://www.kaggle.com/) $\rightarrow$ Chọn **Create New Notebook**.
2. Tại bảng điều khiển bên phải (*Notebook Settings*):
   * **Accelerator**: Chọn **GPU T4 x2** (hoặc GPU T4 x1).
   * **Internet**: Bật **On**.

### Bước 2: Chuẩn bị mã nguồn trên Kaggle
Trong Cell code đầu tiên:
```python
# 1. Cài đặt các thư viện cần thiết
!pip install -q av python_speech_features onnxruntime

# 2. Upload hoặc clone thư mục VN-AV-DF-Active-Speaker vào /kaggle/working/
%cd /kaggle/working/VN-AV-DF-Active-Speaker

# 3. Tải nhanh trọng số mô hình
!python tools/download_weights.py
```

### Bước 3: Chạy Benchmark trực tiếp trên GPU Kaggle
```python
!python tools/evaluate_poc.py --manifest data/poc_manifest.csv --out_dir /kaggle/working/poc_eval
```

### Bước 4: Hiển thị trực tiếp kết quả Dashboard và Video trên Notebook
```python
from IPython.display import Image, display

# Xem ảnh Diagnostic Dashboard của clip kiểm thử
sample_id = "5CNS_gaH16c_s0000653218_e0000658270"
display(Image(f'/kaggle/working/poc_eval/{sample_id}/{sample_id}_diagnostic_dashboard.png'))
```

---

## 6. Định Dạng Dữ Liệu Đầu Ra (Output Data Contracts)

Mỗi clip sau khi xử lý sẽ xuất ra 2 file cấu trúc JSON chuẩn:

### 1. File `result.json` (Curation Decision Metadata)
```json
{
  "clip_id": "5CNS_gaH16c_s0000653218_e0000658270",
  "decision": "manual",
  "reason": "ambiguous",
  "voiced_ms": 5000,
  "visible_active_speech_ratio": 0.72,
  "longest_unexplained_speech_ms": 600,
  "primary_speaker_track_id": 0,
  "early_exit_tier": 2
}
```

### 2. File `active_roi.json` (Downstream Face Handover for Deepfake Models)
```json
{
  "clip_id": "5CNS_gaH16c_s0000653218_e0000658270",
  "primary_speaker_track_id": 0,
  "roi_trajectory": [
    {
      "frame": 0,
      "is_speaking": true,
      "raw_bbox": [838.5, 252.0, 1291.5, 705.0],
      "padded_roi": [725, 139, 1405, 818],
      "track_id": 0
    }
  ]
}
```

---

## 7. Ghi Chú Bản Phát Hành (Release Notes)

> [!NOTE]  
> **Phiên bản hiện tại: `v1.0-poc` (First Working Version)**  
> * Bản phát hành này tập trung hoàn thiện toàn bộ **Source Code thực thi lõi**, **Kiến trúc 3-Tier Cascade**, **Bộ giải mã độc lập PyAV**, và **Hệ thống Diagnostic Dashboard 5 tầng**.  
> * **Mục (3) Hướng dẫn Thu thập & Đọc Kết quả Đánh giá Performance sau khi chạy** (báo cáo phân tích chuyên sâu về Precision, Recall, Rejection Rate thực tế trên tập dữ liệu đầy đủ) sẽ được nhóm hoàn thiện và bổ sung vào tài liệu sau khi hoàn thành đợt chạy thực nghiệm tổng thể.
