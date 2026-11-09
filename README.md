# CAM-Guided Medical Counterfactual Image Generation

## Tổng quan

Dự án nghiên cứu **tạo sinh ảnh phản thực X-quang ngực có hướng dẫn bởi Grad-CAM** nhằm giải thích quyết định của mô hình phân loại ảnh **bình thường (normal)** và **viêm phổi (pneumonia)**. Ảnh phản thực được tạo bằng cách điều chỉnh biểu diễn tiềm ẩn để mô hình chuyển sang dự đoán lớp đối lập, đồng thời hạn chế thay đổi ngoài vùng quan trọng.

Hai notebook thực hiện thí nghiệm đối chứng để đánh giá ảnh hưởng của ràng buộc ngoài vùng quan tâm (ROI):

| Notebook | `LAMBDA_OUT` | Vai trò |
| --- | ---: | --- |
| [loss_out_0.ipynb](loss_out_0.ipynb) | 0 | Bỏ đóng góp của loss ngoài ROI vào hàm mục tiêu và tiêu chí chọn ảnh. |
| [loss_out_10.ipynb](loss_out_10.ipynb) | 10 | Phạt thay đổi ngoài ROI để khuyến khích bảo toàn các vùng này. |

Mã nguồn và các cấu hình còn lại giống nhau; cả hai vẫn tính, lưu và hiển thị sai khác ngoài ROI.

## Phương pháp và cấu hình

Quy trình: **Ảnh gốc → ReStyle-pSp → latent W+ → StyleGAN2 tái tạo → Grad-CAM xác định ROI → tối ưu W+ → ảnh phản thực và đánh giá**.

- **Dữ liệu:** ảnh grayscale 256 × 256 từ [bộ dữ liệu Chest X-ray trên Hugging Face](https://huggingface.co/datasets/Purino/chest-xray-pneumonia-256-fullframe-gray). Hai lần chạy đã lưu sử dụng cùng **4.000/5.216 ảnh thuộc split `train`**, gồm 1.341 ảnh normal và 2.659 ảnh pneumonia, chọn xen kẽ theo lớp với seed 42.
- **Mô hình:** [ReStyle-pSp tại step 3.500](https://huggingface.co/Purino/restyle-psp-chest-xray-pneumonia-256-gray) thực hiện 5 vòng mã hóa; [StyleGAN2-ADA tại 500 kimg](https://huggingface.co/Purino/stylegan2-ada-chest-xray-pneumonia-256-gray-scratch) sinh ảnh; [ResNet50 full-frame](https://huggingface.co/Purino/resnet50-chest-xray-256-fullframe-gray-v2) phân loại. Các notebook sử dụng checkpoint có sẵn, cố định trọng số mô hình và noise của generator, chỉ cập nhật W+.
- **ROI:** chọn khoảng 30% pixel có điểm Grad-CAM cao nhất của lớp dự đoán trên ảnh gốc; mask giữ cố định trong quá trình tối ưu. Classifier quan sát toàn khung ảnh.
- **Tối ưu:** Adam, 300 bước, learning rate 0,01; dùng trực tiếp latent từ encoder (`REFINE_RECONSTRUCTION=False`). Thử cả hai hướng normal → pneumonia và pneumonia → normal.

Hàm mục tiêu gồm ba thành phần:

$$\mathcal{L}=\lambda_{out}\mathcal{L}_{out}+\mathcal{L}_{cls}+10^{-4}\mathcal{L}_{latent}$$

Trong đó, `L_out` là sai khác L1 ngoài ROI so với ảnh gốc; `L_cls` là cross-entropy với lớp đích; `L_latent` là tổng bình phương độ lệch của W+ so với latent ban đầu. Ràng buộc ngoài ROI là **ràng buộc mềm**, không bảo đảm pixel được giữ nguyên tuyệt đối.

## Kết quả thực nghiệm

Các số liệu dưới đây lấy từ **output đã lưu trong hai notebook**. Mỗi lần chạy hoàn tất 4.000 ảnh; **3.861 ảnh (96,53%) đủ điều kiện đánh giá** vì ảnh gốc được phân loại đúng và ảnh tái tạo giữ nguyên lớp. Có 139 ảnh bị loại: 73 ảnh gốc phân loại sai và 66 ảnh tái tạo đổi lớp.

| Chỉ số tổng thể | `λ_out = 0` | `λ_out = 10` |
| --- | ---: | ---: |
| Số ảnh đủ điều kiện | 3.861 | 3.861 |
| Số ảnh đổi sang lớp đích | 3.836 | 3.844 |
| Validity — tỷ lệ đổi lớp | 99,35% | **99,56%** |
| Confidence drop so với ảnh gốc, Mean ± SD | 0,43491 ± 0,08282 | **0,86990 ± 0,11411** |
| Confidence drop so với ảnh tái tạo, Mean ± SD | 0,40929 ± 0,08716 | 0,84429 ± 0,11890 |
| MAE ngoài ROI, thấp hơn là tốt hơn | 0,14613 | **0,06359** |

Kết quả theo từng hướng chuyển lớp:

| Hướng chuyển lớp | Số ảnh đủ điều kiện | Validity (`λ=0` / `λ=10`) | Confidence drop so với ảnh gốc (`λ=0` / `λ=10`), Mean ± SD |
| --- | ---: | ---: | --- |
| normal → pneumonia | 1.319 | 99,17% / 99,39% | 0,51615 ± 0,05802 / 0,95716 ± 0,07928 |
| pneumonia → normal | 2.542 | 99,45% / 99,65% | 0,39275 ± 0,05889 / 0,82463 ± 0,10255 |

**Diễn giải:** trong hai lần chạy này, `λ_out = 10` giảm khoảng **56,5% MAE ngoài ROI**, tăng nhẹ tỷ lệ đổi lớp và làm giảm xác suất lớp gốc nhiều hơn. Kết quả cho thấy ràng buộc ngoài ROI hữu ích trong cấu hình đang khảo sát.

Validity được tính trên toàn bộ 3.861 ảnh đủ điều kiện, bao gồm ca lỗi tối ưu. Log ghi nhận 16 ca `numerical_failure` ở mỗi lần chạy; số ca `target_not_reached` lần lượt là 9 và 1. Confidence drop là chênh lệch xác suất lớp gốc trước và sau tạo sinh; SD là độ lệch chuẩn mẫu giữa các ảnh (`ddof=1`). MAE ngoài ROI được tính trên tensor đầu ra thô ở thang `[-1, 1]`.

**Phạm vi kết luận:** đây là kết quả trên tập con của **train**, chưa phải đánh giá trên tập kiểm thử độc lập hay qua nhiều lần chạy. Một số mô tả trong notebook còn ghi `test`; README này dùng cấu hình và log thực tế. Việc đổi dự đoán classifier chưa chứng minh ảnh phản thực đúng về mặt lâm sàng hoặc mô hình đã khái quát tốt sang dữ liệu mới.

## Chạy thử và đầu ra

Mở notebook trên **Kaggle**, chọn **GPU T4 ×2**, bật Internet và chạy các cell theo thứ tự. Notebook tự thiết lập môi trường Python/PyTorch và tải dữ liệu, checkpoint theo revision cố định. Với cấu hình lưu lên Hugging Face hiện tại, cần secret `HF_TOKEN` có quyền ghi vào repo kết quả; có thể đặt `HF_CHECKPOINT_EVERY=0` và bỏ qua phần upload nếu chỉ lưu cục bộ. Chỉnh `MAX_IMAGES` để chạy thử ít ảnh hơn.

Đầu ra gồm báo cáo JSON/CSV, chỉ số từng ảnh, ảnh gốc/tái tạo/phản thực, bản đồ CAM và mask ROI, latent W+, tensor float và lịch sử loss. Kết quả được lưu dưới `/kaggle/working/chest_cf`, hỗ trợ tiếp tục sau gián đoạn và đồng bộ lên repo `Purino/chest-xray-counterfactual-cam-results`. Tensor float dùng để tái đánh giá; PNG dùng để minh họa.
