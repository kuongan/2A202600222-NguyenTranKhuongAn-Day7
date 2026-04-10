# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Trần Khương An
**Nhóm:** C401-X2
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
 High cosine similarity nghĩa là hai vector chỉ về cùng một hướng trong không gian đa chiều, cho thấy nội dung hoặc ngữ nghĩa của chúng rất tương đồng nhau, bất kể độ dài của văn bản.

**Ví dụ HIGH similarity:**
- **Sentence A:** "Tôi rất thích ăn phở bò."
- **Sentence B:** "Món phở bò là món ăn yêu thích của tôi."
- **Tại sao tương đồng:** Cả hai câu đều chia sẻ cùng một ngữ cảnh, các từ khóa chính và ý nghĩa cốt lõi về sở thích ẩm thực.

**Ví dụ LOW similarity:**
- **Sentence A:** "Thị trường chứng khoán hôm nay biến động mạnh."
- **Sentence B:** "Cách làm bánh mì tại nhà rất đơn giản."
- **Tại sao khác:** Hai câu thuộc hai chủ đề hoàn toàn khác nhau (tài chính và ẩm thực), không có từ ngữ hay ý nghĩa chung.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
Vì Cosine similarity tập trung vào **hướng** (ngữ nghĩa) thay vì **độ dài** của vector; điều này giúp so sánh công bằng giữa một đoạn văn ngắn và một văn bản dài có cùng nội dung.

---

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**


*Trình bày phép tính:*

Sử dụng công thức: $N = \lceil \frac{L - O}{S - O} \rceil$

Trong đó:

* $L$ (Tổng độ dài) = 10,000

* $S$ (Chunk size) = 500

* $O$ (Overlap) = 50
>

Phép tính: $\frac{10,000 - 50}{500 - 50} = \frac{9,950}{450} \approx 22.11$
>

*Đáp án:* **23 chunks** (Làm tròn lên để bao phủ phần dư cuối cùng).

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

Khi overlap tăng, số lượng chunks sẽ **tăng lên** . Chúng ta muốn overlap nhiều hơn để đảm bảo ngữ cảnh giữa các đoạn không bị cắt đứt, giúp mô hình AI hiểu được mối liên kết thông tin giữa các chunk liền kề.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Vietnamese Disease 

**Tại sao nhóm chọn domain này?**
Nhóm chọn domain bệnh học tiếng Việt vì dữ liệu giàu thuật ngữ chuyên ngành và có cấu trúc nội dung lặp lại (triệu chứng, nguyên nhân, điều trị), rất phù hợp để đánh giá chất lượng chunking và retrieval. Ngoài ra, nhu cầu tìm kiếm thông tin y tế trong tiếng Việt là thực tế và giúp kiểm thử tính hữu ích của hệ thống RAG trong bối cảnh người dùng phổ thông.

### Data Inventory


| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | alzheimer.md| https://tamanhhospital.vn/alzheimer/| 27966 | source: "https://tamanhhospital.vn/alzheimer/", category: "bệnh thoái hóa thần kinh"|
| 2 | benh-san-day.md| https://tamanhhospital.vn/benh-san-day/ | 12700 | source: "https://tamanhhospital.vn/benh-san-day/", category: "bệnh ký sinh trùng" |
| 3 | benh-tri.md| https://tamanhhospital.vn/benh-tri/ | 12569 | source: "https://tamanhhospital.vn/benh-tri/", category: "bệnh lí hậu môn - trực tràng" |
| 4 | benh-dai.md| https://tamanhhospital.vn/benh-dai/ | 12700 | source: "https://tamanhhospital.vn/benh-dai/", category: "bệnh truyền nhiễm" |
| 5 | benh-lao-phoi.md| https://tamanhhospital.vn/benh-lao-phoi/ | 12704 | source: "https://tamanhhospital.vn/benh-lao-phoi/", category: "bệnh truyền nhiễm" |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "https://tamanhhospital.vn/benh-tri/" | Dùng để trích dẫn nguồn (citation) trong câu trả lời của AI và giúp người dùng kiểm chứng thông tin gốc. |
| category | string | bệnh ký sinh trùng | Cho phép lọc (filter) nhanh các nhóm bệnh cụ thể, thu hẹp phạm vi tìm kiếm khi người dùng hỏi về một loại bệnh lý nhất định. |


---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tai lieu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| alzheimer.md | FixedSizeChunker (fixed_size) | 137 | 199.99 | Medium |
| alzheimer.md | SentenceChunker (by_sentences) | 74 | 368.93 | High |
| alzheimer.md | RecursiveChunker (recursive) | 1254 | 20.71 | Low |
| benh-dai.md | FixedSizeChunker (fixed_size) | 61 | 197.46 | Medium |
| benh-dai.md | SentenceChunker (by_sentences) | 29 | 414.00 | High |
| benh-dai.md | RecursiveChunker (recursive) | 568 | 20.07 | Low |
| benh_lao_phoi.md | FixedSizeChunker (fixed_size) | 61 | 199.15 | Medium |
| benh_lao_phoi.md | SentenceChunker (by_sentences) | 27 | 448.74 | High |
| benh_lao_phoi.md | RecursiveChunker (recursive) | 412 | 28.32 | Low |


### Strategy Của Tôi

**Loại:** Customer strategy (MarkdownHeaderChunker)


**Mô tả cách hoạt động:**
> Chunk phần context header 3 (###), header 1,2 được tính là metadata của header 3 

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu của nhóm là về bệnh, các tài liệu đều có chung cấu trúc với header 1 là tên bệnh, header 2 là các phần lớn trong mục lục gồm các header 3, header 3 chỉ chứa các phần text ngắn có nội dung thống nhất

**Code snippet (nếu custom):**
```python
class MarkdownHeaderhunker:
    """
    Chunker chuyên dụng cho tài liệu y tế:
    - Chia nhỏ văn bản dựa trên tiêu đề cấp 3 (###).
    - Lưu trữ Header 1 và Header 2 vào Metadata để giữ ngữ cảnh 'Mẹ - Con'.
    - Tích hợp Source và Category từ tài liệu gốc.
    """

    def __init__(self, source: str, category: str):
        self.source = source
        self.category = category
        # Định nghĩa các cấp độ tiêu đề
        self.header_mapping = {
            "#": "header_1",
            "##": "header_2",
            "###": "header_3"
        }

    def chunk(self, text: str) -> list[dict]:
        lines = text.split("\n")
        chunks = []
        
        # Khởi tạo trạng thái Metadata hiện tại
        current_metadata = {
            "source": self.source,
            "category": self.category,
            "header_1": "",
            "header_2": "",
            "header_3": ""
        }
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Kiểm tra xem dòng có phải là tiêu đề (Markdown Header) không
            header_match = re.match(r'^(#{1,3})\s+(.*)', line)
            
            if header_match:
                # Nếu gặp tiêu đề mới và đã có nội dung tích lũy, lưu chunk cũ lại
                if current_content:
                    chunks.append({
                        "content": "\n".join(current_content),
                        "metadata": current_metadata.copy()
                    })
                    current_content = []

                header_level = header_match.group(1) # #, ##, hoặc ###
                header_text = header_match.group(2)
                
                # Cập nhật metadata theo cấp độ tiêu đề
                if header_level == "#":
                    current_metadata["header_1"] = header_text
                    current_metadata["header_2"] = "" # Reset con khi mẹ thay đổi
                    current_metadata["header_3"] = ""
                elif header_level == "##":
                    current_metadata["header_2"] = header_text
                    current_metadata["header_3"] = "" # Reset cháu khi cha thay đổi
                elif header_level == "###":
                    current_metadata["header_3"] = header_text
            else:
                # Nếu là dòng văn bản bình thường, thêm vào nội dung của chunk hiện tại
                current_content.append(line)

        # Lưu đoạn cuối cùng sau khi kết thúc vòng lặp
        if current_content:
            chunks.append({
                "content": "\n".join(current_content),
                "metadata": current_metadata.copy()
            })

        return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| alzheimer.md | best baseline (by_sentences) | 74 | 368.93 | High |
| alzheimer.md | **cua toi** (by_sentences) | 74 | 368.93 | High |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Trần Khương An | MarkdownChunker  | 8 | Tách có cấu trúc, độ dài vừa phải, phù hợp data | Bị phụ thuộc vào file markdown |
| Trần Vọng Triền | SentenceChunker  | 6 | Ổn định, dễ điều chỉnh | không giữ được long-term dependency  |
| Phương Hoàng Yến | MarkdownHeadChunker | 8/10 | Từng vector là nội dung ngắn, thống nhất. Kèm với header 1,2 trong metadata giúp hệ thống retrieval có đủ thông tin từ đề mục | Phụ thuộc hoàn toàn vào định dạng gốc. |
| Phạm Minh Việt | LateChunker | 8 | Nhieu ngu canh | Chunk dai hon |
| Lê Hoàng Minh | LateChunker | 8 | Nhieu ngu canh | Chunk dai hon |
| Nguyễn Minh Châu | SentenceChunker | 6.2 | Ổn định, dễ điều chỉnh | không giữ được long-term 

**Strategy nào tốt nhất cho domain này? Tại sao?**
> MarkdownHeadChunker phù hợp nhất vì nó giữ trọn vẹn cấu trúc phân cấp của dữ liệu dạng markdown.
---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Su dung regex `(?<=[.!?])\s+` de tach cau theo dau cau va khoang trang. Loai bo khoang trang thua va cau rong. Giu lai ky tu dau cau de cau khong bi mat nghia.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thu separator theo thu tu uu tien, neu doan con dai thi de quy voi separator tiep theo. Base case: doan nho hon chunk_size hoac het separator thi cat theo kich thuoc. Muc tieu la giu ranh gioi tu nhien neu co the.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` embed content va luu record gom doc_id, content, metadata, embedding. `search` embed query, tinh dot product voi cac embedding va sap xep giam dan de lay top_k. Neu dung Chroma, dung collection.query voi query_embeddings.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` loc metadata truoc, sau do moi chay similarity search de dam bao precision. `delete_document` xoa tat ca chunk theo doc_id (in-memory: loc list; Chroma: get ids va delete).

### KnowledgeBaseAgent

**`answer`** — approach:
> `answer` lay top_k chunks, ghep thanh context va chen vao prompt theo dang: Context -> Question -> Answer. LLM tra loi dua tren context, giam hallucination. Cac chunk duoc noi bang newline de doc de dang.

### Test Results
```
# Paste output of: pytest tests/ -v
```

```
============================================================= test session starts =============================================================
platform win32 -- Python 3.13.12, pytest-9.0.3, pluggy-1.6.0 -- c:\Users\MSI\vinuni\Day-07-Lab-Data-Foundations\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\MSI\vinuni\Day-07-Lab-Data-Foundations
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                    [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                             [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                      [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                       [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                            [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                            [ 14%] 
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                  [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                   [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                 [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                   [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                   [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                              [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                          [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                    [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                           [ 35%] 
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                               [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                         [ 40%] 
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                               [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                   [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                     [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                       [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                             [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                  [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                    [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                        [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                     [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                              [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                             [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                        [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                    [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                               [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                   [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                         [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                   [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                              [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                             [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                 [ 90%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                            [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                     [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                           [ 97%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                               [100%] 

============================================================= 42 passed in 0.34s ============================================================== 
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)


| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
| :--- | :--- | :--- | :--- | :---: | :---: |
| 2 | Vi khuẩn Mycobacterium tuberculosis là tác nhân chính gây ra bệnh lao phổi lây nhiễm. | Schistosoma là loại ký sinh trùng gây ra căn bệnh truyền nhiễm có tên là bệnh đái. | low | 0.25 | Yes |
| 1 | Chứng thoái hóa thần kinh gây ra tình trạng mất trí nhớ được gọi là Alzheimer. | Sự suy giảm về hành vi và chức năng nhận thức là hệ quả của bệnh Alzheimer. | high | 0.85 | Yes |
| 4 | Alzheimer là một căn bệnh về não bộ với biểu hiện đặc trưng là suy giảm trí nhớ. | Chỉ số chứng khoán ghi nhận mức sụt giảm nghiêm trọng trong phiên giao dịch hôm nay. | low | 0.12 | Yes |
| 3 | Các triệu chứng về đường tiêu hóa do bệnh sán dây gây ra rất phổ biến tại khu vực nhiệt đới. | Tình trạng suy dinh dưỡng và thiếu máu có thể bắt nguồn từ việc nhiễm sán dây. | high | 0.78 | Yes |
| 5 | Những cơn đau và hiện tượng xuất huyết tại vùng hậu môn là dấu hiệu của bệnh trĩ. | Bệnh nhân mắc trĩ có thể lựa chọn phương pháp can thiệp ngoại khoa hoặc dùng thuốc. | high | 0.82 | Yes |


**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
>  Kết quả bất ngờ nhất là cặp số 3 vì mô hình nhận diện được sự tương đồng sâu giữa địa lý và lâm sàng thay vì chỉ khớp từ vựng. Điều này chứng tỏ embeddings biểu diễn nghĩa bằng cách ánh xạ các khái niệm vào không gian đa chiều dựa trên mối liên hệ thực thể, thay vì chỉ so sánh bề nổi của ngôn ngữ.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Bệnh trĩ có ảnh hưởng khả năng sinh sản không. | Không |
| 2 | Ăn cá có bị sán không | Có. Có loại sán dây ở bên trong cá. Có khả năng lây bệnh cho người |
| 3 | Làm sao biết mình bị Alzheimer: | - Sa sút trí nhớ và khả năng nhận thức<br>- Khó khăn diễn đạt bằng ngôn ngữ<br>- Thay đổi hành vi, tâm trạng, tính cách<br>- Nhầm lẫn thời gian hoặc địa điểm<br>- Đặt đồ vật sai vị trí và không thể nhớ lại mình đã từng làm gì |
| 4 | Những đối tượng nào có nguy cơ cao chuyển từ tình trạng lao tiềm ẩn sang bệnh lao phổi (lao bệnh)? | - Người nhiễm HIV.<br>- Người sử dụng ma túy dạng chích.<br>- Người bị sụt cân (khoảng 10%).<br>- Bệnh nhân mắc bệnh bụi phổi silic, suy thận hoặc đang chạy thận, và bệnh đái tháo đường.<br>- Người từng thực hiện phẫu thuật cắt dạ dày hoặc ruột non.<br>- Người ghép tạng hoặc đang sử dụng thuốc corticoid kéo dài, thuốc ức chế miễn dịch.<br>- Bệnh nhân bị ung thư đầu cổ. |
| 5 | Trong trường hợp bị động vật cắn hoặc cào xước, quy trình sơ cứu tại chỗ và các biện pháp y tế cần thực hiện ngay lập tức là gì để ngăn chặn virus dại xâm nhập hệ thần kinh? | **Dựa theo hướng dẫn, quy trình xử lý bao gồm:**<br><br>**1. Sơ cứu tại chỗ:**<br>- Rửa ngay vết thương rộng bằng nước sạch và xà phòng, chất tẩy rửa, povidone iodine... trong ít nhất 15 phút.<br>- Sát trùng vết thương bằng cồn 70% (ethanol) hoặc povidone-iodine.<br>- Băng bó đơn giản và đưa đến cơ sở y tế.<br><br>**2. Can thiệp y tế:**<br>- Bác sĩ thăm khám và chỉ định tiêm vắc xin phòng dại càng sớm càng tốt.<br>- Có thể chỉ định tiêm thêm huyết thanh kháng dại.<br><br>**3. Theo dõi động vật:**<br>- Theo dõi con vật đã cắn (cắn không lý do, tiết nước bọt quá mức, chết sau vài ngày...). |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Bệnh trĩ có ảnh hưởng khả năng sinh sản không?| Bệnh trĩ: Dấu hiệu nguyên nhân cách phòng, có lây và nguy hiểm không? > **Những câu hỏi thường gặp về bệnh trĩ** > **1. Bệnh trĩ có ảnh hưởng đến sinh sản không?** Nội dung: Xin trả lời, trĩ là bệnh lý về tiêu hóa vì vậy không gây ảnh hưởng đến khả năng sinh sản| 0.6936 | Yes |  Bệnh trĩ: Dấu hiệu nguyên nhân cách phòng, có lây và nguy hiểm không? > **Những câu hỏi thường gặp về bệnh trĩ** > **1. Bệnh trĩ có ảnh hưởng đến sinh sản không?** Nội dung: Xin trả lời, trĩ là bệnh lý về tiêu hóa vì vậy không gây ảnh hưởng đến khả năng sinh sản|
| 2 | Ăn cá có bị sán không? |Đoạn nội dung: Ngữ cảnh: Bệnh sán dây lây nhiễm cho người qua đường nào? Dấu hiệu, nguyên nhân > Nguyên nhân gây nhiễm sán dây > Ăn thịt hoặc cá bị nhiễm bệnh Nội dung: Nếu thịt hoặc cá mang nang ấu trùng và không được nấu chín hoặc còn sống, thì những nang này có thể đi đến ruột và phát triển thành sán dây trưởng thành. Một con sán dây trưởng thành có thể có các đặc điểm sau: Sống lâu đến 20 năm Dài tới 15m Tự gắn vào thành ruột Đi qua hệ thống tiêu hóa của con người và đào thải qua phân Nhiễm sán dây cá phổ  | 0.5044|Yes |Bệnh sán dây lây nhiễm cho người qua đường nào? Dấu hiệu, nguyên nhân > Nguyên nhân gây nhiễm sán dây > Ăn thịt hoặc cá bị nhiễm bệnh Nội dung: Nếu thịt hoặc cá mang nang ấu trùng và không được nấu chín hoặc còn sống, thì những nang này có thể đi đến ruột và phát triển thành sán dây trưởng thành. Một con sán dây trưởng thành có |
| 3 | Làm sao biết mình bị Alzheimer? | Ngữ cảnh: Bệnh Alzheimer: Nguyên nhân, triệu chứng, điều trị và phòng ngừa > **Bệnh Alzheimer là gì?** Nội dung: **Alzheimer** là một căn bệnh gây ra tình trạng mất trí nhớ, mất các chức năng nhận thức, làm ảnh hưởng nhiều đến chất lượng sống và làm việc của người bệnh. Tuy nhiên đây không phải là sự lão hóa bình thường, vì vậy đừng nhầm lẫn Alzheimer với hiện tượng suy giảm trí nhớ thông thường ở người già. (1) Có một ngày bạn bỗng thấy ông, bà, cha, mẹ,… càng có tuổi sẽ càng trở nên khó tính | 0.7108| No | Bệnh Alzheimer: Nguyên nhân, triệu chứng, điều trị và phòng ngừa > **Bệnh Alzheimer là gì?** Nội dung: **Alzheimer** là một căn bệnh gây ra tình trạng mất trí nhớ, mất các chức năng nhận thức, làm ảnh hưởng nhiều đến chất lượng sống và làm việc của người bệnh. Tuy nhiên đây không phải là sự lão hóa bình thường, vì vậy đừng nhầm |
| 4 | Những đối tượng nào có nguy cơ cao chuyển từ tình trạng lao tiềm ẩn sang bệnh lao phổi (lao bệnh)?|Ngữ cảnh: Bệnh lao phổi (ho lao): Nguyên nhân, dấu hiệu và cách điều trị > **Đối tượng có nguy cơ mắc bệnh** Nội dung: Lao dễ lây từ người sang người qua đường hô hấp, vì thế những đối tượng sau đây có nguy cơ cao mắc lao phổi: Người có tiếp xúc, nói chuyện, chăm sóc gần gũi với người mắc bệnh lao Người sống và làm việc tại vùng có tỷ lệ mắc lao cao, hay nơi có bệnh nhân lao sinh sống Người bị mắc các bệnh gây suy giảm miễn dịch như HIV, bệnh gan, lách |0.7345 |Yes | Nội dung: Lao dễ lây từ người sang người qua đường hô hấp, vì thế những đối tượng sau đây có nguy cơ cao mắc lao phổi: Người có tiếp xúc, nói chuyện, chăm sóc gần gũi với người mắc bệnh lao Người sống và làm việc tại vùng có tỷ lệ |
| 5 | Trong trường hợp bị động vật cắn hoặc cào xước, quy trình sơ cứu tại chỗ và các biện pháp y tế cần thực hiện ngay lập tức là gì để ngăn chặn virus dại xâm nhập hệ thần kinh?|Bệnh dại: Triệu chứng, nguyên nhân, chẩn đoán và phòng ngừa > **Các câu hỏi thường gặp** > **1. Tôi bị động vật cắn trầy xước phải làm gì?** Nội dung: Khi bị động vật cắn, nhất là chó cắn, điều đầu tiên bạn nên làm là rửa vết thương bằng xà phòng dưới vòi nước sạch ngay lập tức. Chỗ vết cắn, trầy xước hãy làm sạch hoàn toàn với cồn 70% rượu/ethanol hoặc povidone-iodine (nếu có). Hoặc có thể làm sạch bằng xà phòng, chất tẩy rửa,… ít nhất 15 phút. Sau đó khử trùng vết thương bằng chất kh |0.5756 |Yes | **1. Tôi bị động vật cắn trầy xước phải làm gì?** Nội dung: Khi bị động vật cắn, nhất là chó cắn, điều đầu tiên bạn nên làm là rửa vết thương bằng xà phòng dưới vòi nước sạch ngay lập tức. Chỗ vết cắn, trầy xước hãy làm sạch hoàn toàn với |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---
## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Học được thêm về các ưu nhược điểm các loại chunker khác và cách thực hiện , module hóa.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Kỹ thuật triển khai RAG evaluation tự động để đo lường độ chính xác và việc áp dụng Metadata filtering để tăng khả năng lọc thông tin chuyên sâu.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ chú trọng tiền xử lý dữ liệu sạch hơn và triển khai Hybrid Search nhằm kết hợp ưu điểm của cả tìm kiếm ngữ nghĩa lẫn từ khóa chính xác cho các thuật ngữ y khoa.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **90 / 100** |
