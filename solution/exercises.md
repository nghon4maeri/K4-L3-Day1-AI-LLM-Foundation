# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Nhiệt độ càng thấp (0.0) phản hồi càng ổn định, lặp lại cùng một cấu trúc, thậm chí gần như giống hệt nhau giữa các lần gọi. Nhiệt độ càng cao (1.5) phản hồi càng đa dạng, sáng tạo hơn về từ ngữ và ví dụ, nhưng dễ lan man hoặc lệch đề. Quy luật chung: temperature điều khiển mức ngẫu nhiên khi chọn token tiếp theo — thấp cho kết quả nhất quán, cao cho kết quả phong phú.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng 0.2–0.3. Vì chatbot hỗ trợ khách hàng cần trả lời chính xác, nhất quán và theo đúng quy trình/chính sách của công ty; nhiệt độ thấp giúp giảm bịa đặt (hallucination), tránh trả lời khác nhau cho cùng một câu hỏi, mang lại trải nghiệm đáng tin cậy. Độ sáng tạo cao không cần thiết cho tác vụ này.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload: 10.000 người × 3 lần × 350 token ≈ 10.500.000 token đầu ra/ngày. GPT-4o: 10.500K × $0.010 = $105/ngày; GPT-4o-mini: 10.500K × $0.0006 = $6,3/ngày → GPT-4o đắt hơn khoảng **16,7 lần**. Nên dùng GPT-4o khi cần suy luận phức tạp, độ chính xác cao như tư vấn pháp lý/y tế, gỡ lỗi code khó, biên tập nội dung tinh tế. Nên dùng mini cho phân loại, trích xuất, FAQ, paraphrase — tác vụ đơn giản, khối lượng lớn, giá rẻ đáng kể.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona "giáo viên tiểu học", phản hồi ngắn gọn, dùng từ đơn giản, so sánh blockchain với trò chơi hoặc sổ ghi nợ, kèm ví dụ quen thuộc với trẻ. Với persona "chuyên gia tài chính", phản hồi dài hơn, dùng thuật ngữ kỹ thuật như sổ cái phân tán, hợp đồng thông minh, đồng thuận. Cùng một câu hỏi nhưng system prompt thay đổi giọng điệu, mức chi tiết, từ vựng và cách lấy ví dụ — nó định hình toàn bộ "nhân cách" và nguyên tắc trả lời của model trong phiên.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Ví dụ đoạn tiếng Việt ~100 từ: `count_tokens` (tiktoken) cho ~155 token, trong khi ước lượng `số từ / 0.75` cho ~133 — chênh khoảng **~16–17%** (tiktoken nhiều hơn). Tiktoken dùng Byte-Pair Encoding hoạt động trên byte: tiếng Việt có dấu nên mỗi từ dài nhiều byte hơn từ tiếng Anh cùng "độ dài từ", và một từ tiếng Việt thường bị tách thành nhiều token; tiếng Anh trung bình ~1,3 token/từ, còn tiếng Việt thường ~1,5–2 token/từ. Vì vậy đếm từ là ước lượng thô, tokenizer mới cho con số khớp cách tính tiền của nhà cung cấp.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất với trải nghiệm hội thoại tương tác — chatbot, copilot, trợ lý viết — nơi câu trả lời dài và người dùng muốn thấy phản hồi xuất hiện ngay (giảm cảm giác chờ đợi, có thể ngắt giữa chừng). Nó cũng hữu ích cho các tác vụ sinh nội dung dài như bài viết, email. Non-streaming phù hợp hơn khi phản hồi ngắn, cần kết quả hoàn chỉnh để xử lý tiếp (như parse JSON, kiểm tra cấu trúc, gọi nội bộ máy-máy, batch) — vì nhận cả response rồi mới xử lý sẽ đơn giản và đáng tin cậy hơn.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng thời gian chờ theo cấp số nhân (0,1s → 0,2s → 0,4s...) nên các client không dồn dập đánh vào server cùng lúc; chúng tự phân tán thời điểm thử lại, cho server thời gian phục hồi. Nếu hàng nghìn client cùng retry với delay cố định giống nhau (ví dụ đều đợi đúng 1s), chúng sẽ "đồng bộ" lại và cùng bắn request vào cùng một thời điểm (thundering herd), khiến server tiếp tục quá tải và lỗi lặp lại theo chu kỳ.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona của tôi: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." Hai lựa chọn từ ngữ quan trọng: (1) "trả lời ngắn gọn" — giới hạn độ dài phản hồi, vừa đúng nhu cầu hỏi–đáp của sinh viên vừa kiểm soát chi phí token; (2) "bằng tiếng Việt" — khóa học dùng tiếng Việt, chỉ định ngôn ngữ tránh model tự ý trả lời tiếng Anh. Cụm "trợ giảng thân thiện" đặt giọng điệu hỗ trợ, kiên nhẫn.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là không có bộ nhớ dài hạn: history chỉ giữ 3 lượt gần nhất, nên trợ lý "quên" thông tin đã nói sớm hơn và cũng không nhớ qua các phiên. Đề xuất: cải thiện bằng cách thêm "bộ nhớ tóm tắt" — mỗi khi history sắp vượt giới hạn, gọi một lần model để tóm tắt hội thoại thành vài câu và đưa phần tóm tắt này vào system prompt (hoặc một message system riêng) của các lượt sau; triển khai đơn giản là lưu summary vào biến, cập nhật sau mỗi N lượt, và ghép `[{"role": "system", "content": persona + summary}]` khi dựng messages.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
