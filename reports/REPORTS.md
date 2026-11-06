# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
   answer:
   "rank": 1,
   "class_id": 468,
   "class_name": "cab",
   "score": 0.510915
   "taxonomy_name": "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   answer: Thông thường record kiểu phân loại ảnh (image classification) nói về cả bức ảnh (image-level label) chứ không phải một vật thể riêng lẻ.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   answer: Danh sách lớp không nên do model tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   answer: Mỗi trường giải quyết một vấn đề khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   answer:Đây là phần cực kỳ quan trọng của guideline annotation. Guideline cần quy định rõ quy tắc chọn lớp chính (primary class).
   Ví dụ cần làm rõ:
   -Cách xác định chủ thể chính
   -Trường hợp nhiều lớp cùng xuất hiện
   -Trường hợp nền và tiền cảnh
   -Trường hợp không chắc chắn
6. Vì sao không được lấy `score` của model làm ground truth?
   answer: score chỉ là dự đoán của model, không phải sự thật của dữ liệu.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1.  Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
    answer:
    "class_name": "oven",
    "score": 0.6343,
    "bbox_xyxy": [489.09, 200.73, 617.81, 344.88],
    "bbox_width": 128.72,
    "bbox_height": 144.15
2.  Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    answer:
    Box của đối tượng oven có:
    Góc trên-trái (top-left): (489.09, 200.73)
    Góc dưới-phải (bottom-right): (617.81, 344.88)
    Chiều rộng: 128.72 pixel
    Chiều cao: 144.15 pixel

3.  Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    answer:
    Với dữ liệu hiện tại chỉ thấy 1 prediction: oven:0.6343
    do đó: nên prediction này vượt cả hai ngưỡng.

4.  Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    answer:
    Khi hạ ngưỡng, hệ thống sẽ giữ lại nhiều box hơn, bao gồm cả các box chất lượng thấp.
    Reviewer phải:
    Loại bỏ false positive
    Model báo có vật nhưng thực tế không có.
    Kiểm tra class đúng hay sai
    Ví dụ:
    bus → truck
    dog → wolf
    Xử lý box trùng lặp
    Nhiều box cùng bao một vật thể.
    Đánh giá chất lượng box
    Box quá rộng.
    Box quá chật.
    Box lệch vị trí.
    Xác nhận các vật thể khó nhìn
    Nhỏ
    Mờ
    Bị che khuất

5.  Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    answer:
    Vẽ bounding box nhỏ nhất có thể nhưng vẫn bao phủ toàn bộ phần nhìn thấy được của đối tượng. Box phải bám sát biên ngoài của vật thể và không bao gồm khoảng nền không cần thiết.
    Ví dụ:
    Đúng
    Box ôm sát lò nướng.
    Box ôm sát bàn, ghế, tủ lạnh.
    Box chứa toàn bộ phần nhìn thấy của vật thể.
    Sai
    Chừa nhiều khoảng trống xung quanh.
    Bao luôn vật thể khác bên cạnh.
    Mở rộng box để đoán phần không nhìn thấy.
6.  Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    answer:
    Từ record được cung cấp:
    JSON
    bbox_xyxy = [489.09, 200.73, 617.81, 344.88]
    image_width = 640

    Box của oven nằm rất gần mép phải ảnh:
    Plain Text
    xmax = 617.81
    image_width = 640

    Nghĩa là lò nướng có thể đang nằm sát biên ảnh và một phần của đối tượng có khả năng bị cắt mép (truncation).
    Tôi sẽ đưa trường hợp này vào guideline dưới dạng quy tắc:
    Nếu vật thể bị cắt bởi mép ảnh nhưng vẫn nhận diện được rõ ràng, hãy gán nhãn cho vật thể đó và vẽ box bao quanh toàn bộ phần còn nhìn thấy trong ảnh. Không được suy đoán hay mở rộng box ra ngoài ảnh.

    Không cần escalation vì:
    Vật thể vẫn nhận dạng được là oven.
    Trường hợp vật bị cắt mép là tình huống phổ biến trong object detection.
    Có thể giải quyết bằng một quy tắc annotation chung áp dụng cho nhiều ảnh.

    Chỉ nên escalate lên Lab Coach nếu:
    Không xác định được đó là vật gì.
    Phần nhìn thấy quá ít để gán nhãn đáng tin cậy.
    Taxonomy không quy định rõ có nên gán nhãn cho đối tượng bị che khuất/cắt mép ở mức đó hay không.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

   "instance_id": "kitchen-006",
   "class_name": "bowl",
   "score": 0.528202,
   "polygon_point_count": 84,
   "polygon_xy":
   [
   29.0,
   303.0
   ],
   [
   28.0,
   304.0
   ],
   [
   26.0,
   304.0
   ],
   [
   25.0,
   305.0
   ],
   [
   23.0,
   305.0
   ]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    Bounding box chỉ cho biết:
    vị trí đối tượng, chiều rộng, chiều cao và luôn có dạng hình chữ nhật.

    Polygon cho biết:
    -đường biên thực tế của vật thể
    -các chỗ cong, lõm, nhô ra
    -phần nào thuộc vật thể và phần nào là nền

    Ví dụ với một cái bát (bowl):
    -Box sẽ bao cả vùng không khí bên ngoài cái bát vì là hình chữ nhật.
    -Polygon có thể bám sát mép cong của cái bát.

    Do đó polygon mô tả hình dạng chính xác hơn box và thường được dùng cho:
    -instance segmentation
    -mask annotation
    -đo diện tích vật thể
    -các bài toán cần biên chính xác.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    instance_id là mã định danh cho một đối tượng cụ thể trong một ảnh.
    Nó giúp:
    -tham chiếu đúng đối tượng khi review
    -sửa annotation
    -QA và audit dữ liệu
    -liên kết box, mask, metadata của cùng một đối tượng
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ polygon bám sát biên nhìn thấy được của vật thể. Đặt đủ số điểm để theo được các đoạn cong hoặc góc cạnh quan trọng. Không được cắt vào bên trong vật thể và không được mở rộng sang nền hoặc vật thể khác.

✅ Đúng
    Mask ôm sát mép của bát.
    Các đoạn cong được biểu diễn bằng đủ điểm polygon.
    Chỉ bao phủ phần thực sự thuộc vật thể.
❌ Sai
    Để khoảng hở lớn giữa mask và vật thể.
    Bao gồm nền xung quanh.
    Đoán thêm phần bị che khuất hoàn toàn.
    Cắt bỏ một phần vật thể đang nhìn thấy.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    Đối tượng: 
        instance_id = kitchen-006
        class_name = bowl
Mép của chiếc bát (bowl) hòa vào mặt bàn hoặc chồng lấn với vật dụng khác như đĩa, ly hoặc khăn trải bàn, khiến ranh giới không hoàn toàn sắc nét.
Trong trường hợp này tôi sẽ ưu tiên dùng guideline thay vì escalation.

Quy tắc guideline:
Khi biên đối tượng hơi mờ nhưng vẫn nhận diện được ranh giới hợp lý bằng mắt thường, annotator phải vẽ mask theo mép nhìn thấy tốt nhất và giữ tính nhất quán với các ảnh khác trong dự án.

Không cần escalation vì:
-Đây là tình huống phổ biến trong segmentation.
-Annotator vẫn nhận ra đối tượng là một cái bát.
-Có thể xử lý bằng quy tắc chung áp dụng cho toàn bộ tập dữ liệu.

Chỉ escalation khi:
-Hai vật thể chồng lấn quá mạnh không thể xác định ranh giới.
-Vật thể bị che phần lớn.
-Nhiều annotator có thể đưa ra các mask rất khác nhau.
Taxonomy hoặc guideline không quy định cách xử lý trường hợp đó.

Khi đó cần hỏi Lab Coach để thống nhất cách gán nhãn cho toàn bộ dự án, tránh mỗi người vẽ theo một cách khác nhau.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                | Annotator làm gì?                             | Reviewer xem gì? |
| --------------------- | --------------------------------------------- | --------------------------------------------------                | -----------------                             | ---------------- |
| Phân loại ảnh         | 1 nhãn cho cả ảnh.                            | Ảnh có nhiều chủ thể cùng lúc, khó xác định lớp chính.            |                                               | Kiểm tra lớp đã chọn                                                                                                                                                                                          có phù hợp với nội 
                                                                                                                                                                                            dung chính của ảnh và 
                                                                                                                                                                                            đúng taxonomy hay không.                 |

| Phát hiện vật thể     | 1 object = 1 box + 1 class.                   | Vật thể sát mép ảnh hoặc bị cắt mép. Trong ảnh kitchen,           | Chọn lớp theo guideline                       |Kiểm tra class đúng 
                                                                                                                                                                                            |chưa, box có ôm sát
                                                                                                                                                                                            |vật thể không, có bỏ
                                                                                                                                                                                            |sót hoặc trùng box không.
                                                                                                                                              (vật thể/chủ đề chính của ảnh),
                                                                                                                                            |  không tự tạo lớp mới ngoài taxonomy.          
                                                                          box của oven nằm gần mép phải (xmax=617.81 trên ảnh rộng 640px),  |
                                                                          có thể gây tranh luận box nên ôm tới đâu.                         |

 Instance segmentation |  1 object = 1 mask/polygon + 1 class.          | Biên vật thể cong, mờ hoặc chạm vật khác. Với bowl,               | Vẽ bounding box sát biên vật thể              | Kiểm tra mask có bám
                                                                                                                                                                                            |sát biên thật không, có
                                                                                                                                            | thấy được, không đoán phần nằm ngoài ảnh.     |ăn vào nền hoặc bỏ sót
                                                                          polygon có 84 điểm cho thấy biên dạng cong                        |                                               |phần vật thể không, các
                                                                          cần nhiều điểm để mô tả chính xác. Chỗ tiếp xúc giữa              |                                               |instance có bị chồng lấn
                                                                          bát và mặt bàn có thể khó xác định ranh giới.                     |Vẽ polygon bám sát phần nhìn thấy của vật      |sai không.
                                                                                                                                            | thể, thêm đủ điểm ở vùng cong, không lấn 
                                                                                                                                            | sang nền.                                     |                  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    Chỉ sử dụng dữ liệu trong phạm vi được cấp phép cho bài lab, không sao chép, chia sẻ, tải xuống hoặc tiết lộ hình ảnh, nhãn và thông tin dự án cho người hoặc hệ thống không được phép truy cập.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    Tôi sẽ dừng việc gán nhãn/xử lý dữ liệu ngay lập tức và báo cho Lab Coach, Giảng viên hoặc người phụ trách dự án theo quy trình của lớp/phòng lab để được hướng dẫn xử lý tiếp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
