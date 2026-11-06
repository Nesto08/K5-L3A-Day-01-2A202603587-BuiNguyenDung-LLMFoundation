# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603587
- **Họ và tên:** Bùi Nguyên Dũng
- **SEED** (in ở ô đầu notebook): 1286

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.827      0.905        0.919   
2. BoW + MLP (deep learning)         0.966   0.878      0.920        0.974   
3a. Encoder (MiniLM) + kNN           0.895   0.959      0.926        0.947   
3b. Encoder (MiniLM) + LogReg        0.949   0.949      0.949        0.947   
3c. Encoder (MiniLM) + MLP head      0.930   0.949      0.939        0.947   
4a. LLM nhỏ zero-shot                0.792   0.950      0.864        0.864   
4b. LLM lớn zero-shot                1.000   0.950      0.974        0.974   
4c. LLM nhỏ few-shot                 0.950   0.950      0.950        0.950   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            3.445      32818.555       0.063   
2. BoW + MLP (deep learning)         5.627      47423.503       0.182   
3a. Encoder (MiniLM) + kNN          75.661         30.134      54.267   
3b. Encoder (MiniLM) + LogReg       77.669         35.181      26.704   
3c. Encoder (MiniLM) + MLP head     76.919         24.058      29.275   
4a. LLM nhỏ zero-shot                0.000          1.027     762.990   
4b. LLM lớn zero-shot                0.000          1.635     571.165   
4c. LLM nhỏ few-shot                 0.000          1.086     753.423   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.158           0.0  
2. BoW + MLP (deep learning)          0.230           0.0  
3a. Encoder (MiniLM) + kNN           97.493           0.0  
3b. Encoder (MiniLM) + LogReg        41.128           0.0  
3c. Encoder (MiniLM) + MLP head      41.058           0.0  
4a. LLM nhỏ zero-shot              2048.891           0.0  
4b. LLM lớn zero-shot              1124.220           0.0  
4c. LLM nhỏ few-shot               1978.305           0.0  

## 3. Dự đoán và thực tế

+ llm_models: Đoán model nhỏ (gpt-6-luna), thực tế model lớn áp đảo (0..974 so với 0.864) do khả năng suy luận ngữ cảnh mạnh hơn.
+ llm_fewshot: Đoán "giảm", thực tế F1 tăng vọt từ 0.864 đến 0.950 nhờ 4 ví dụ định hình văn phong.   
+ lr=50 (hoặc loss): Đoán nổ phân kỳ hoặc loss cao, thực tế hàm Sigmoid bão hòa ép loss về sát 0 (0.006)

## 4. Phân tích 50 case khó

-Model tụt nhiều nhất: BoW (Logistic Regression/MLP).   
-Nhóm tụt nặng nhất: Né từ khóa (obfuscation) và Prompt injection.   
-2 case cụ thể & cơ chế:
  +Case ngắt ký tự (f.r.e.e): BoW coi là từ lạ nên không kích hoạt trọng số spam.   
  +Case Prompt injection: LLM đọc hiểu như người nên bị câu chỉ thị giả mạo dắt mũi bỏ qua nhiệm vụ phân loại

## 5. Thử thách cá nhân

Đề bài: Đề #3 - Đổi lớp ẩn MLP thành 4 và 512 nơ-ron, báo cáo số tham số, F1 test, thời gian train và lý do chọn mạng nhỏ. 
Cách làm: Khởi tạo mô hình với hidden_dim lần lượt là 4 và 512, huấn luyện 30 epochs và đo thời gian thực tế kèm F1 test. 
Kết quả:
-	Ẩn 4 nơ-ron: 8.009 tham số, train 8.95s, F1 test 0.9200. 
-	Ẩn 512 nơ-ron: 1.025.025 tham số, train 61.00s, F1 test 0.9263. 
Giải thích: Mạng 512 nơ-ron tăng gấp 128 lần tham số và train lâu gấp 7 lần nhưng F1 chỉ nhích thêm 0.0063 do Bag-of-Words thiếu ngữ cảnh sâu. Nên chọn mạng nhỏ khi cần tối ưu bộ nhớ, đáp ứng độ trễ thời gian thực dưới 1 ms hoặc tránh overfitting khi tập train nhỏ. 

## 6. Phương án cho sản phẩm thật

Phương án: Sử dụng kiến trúc Router 2 tầng phối hợp BoW + MLP và LLM. 
Số liệu:
-	BoW + MLP đạt chuẩn SLA với p95 chỉ 0.230 ms (ngưỡng yêu cầu < 50 ms) và F1 test cao nhất trong nhóm offline là 0.920. 
-	Dùng LLM hoàn toàn cho 10 triệu tin/ngày sẽ ngốn ít nhất 3.150 USD/tháng và độ trễ p95 lên tới 1.100 - 2.000 ms, vi phạm hoàn toàn SLA mạng viễn thông. 
Đánh đổi: BoW + MLP giải quyết tức thì 90 - 95% tin nhắn chắc chắn, chỉ 5 - 10% tin phân vân mới gửi lên LLM. Đánh đổi này giúp tiết kiệm hơn 90% chi phí API và đạt SLA cho gần như toàn bộ lưu lượng, chấp nhận rủi ro nhỏ nếu có ca đối kháng tinh vi lọt qua tầng 1

## 7. AI log và điều đã học

AI Log:
-Từ Phần 1 đến Phần 5: Hỏi AI gợi ý hướng triển khai các TODO code và giải thích bản chất các câu hỏi lý thuyết.
-AI phản hồi: Đưa ra gợi ý giải thuật, chỉ lỗi sai logic và hướng dẫn phân tích số liệu/biểu đồ.
-Bản thân kiểm chứng: Tự sửa code, đối chiếu lại công thức, chạy thực nghiệm trên notebook để kiểm tra kết quả và tự viết lại câu trả lời theo cách hiểu của mình.

Điều hiểu ra:
-Hàm Sigmoid bị bão hòa ở learning rate lớn sẽ triệt tiêu gradient và khóa cứng trọng số, khiến loss bị ép sát về 0 thay vì nổ phân kỳ.
-Bag-of-Words chỉ đếm token rời rạc nên bị mù hoàn toàn trước các kỹ thuật ngắt ký tự đối kháng, trong khi Embedding nắm bắt ngữ nghĩa tốt hơn nhưng dễ nhầm tin bàn luận về spam.
-Trong bài toán sản phẩm thực tế, chất lượng F1 chỉ là một nửa câu chuyện; độ trễ SLA và chi phí vận hành mới là yếu tố quyết định để chọn kiến trúc lai Router.
