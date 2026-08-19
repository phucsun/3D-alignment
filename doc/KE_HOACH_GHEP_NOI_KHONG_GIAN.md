# Kế hoạch tối ưu bài toán GHÉP NỐI HAI KHÔNG GIAN 3D qua một bức tường chung

> Tài liệu này tổng hợp toàn bộ phần phản biện giữa Claude và Codex, giải thích **dễ hiểu** (hạn chế thuật ngữ tiếng Anh) về:
> 1. Bài toán là gì, **vì sao khó** (kèm ví dụ từ chính dữ liệu của bạn).
> 2. **Ý tưởng cốt lõi** để giải một cách tổng quát và mạnh mẽ.
> 3. **Kế hoạch chi tiết từng bước** — từ dữ liệu thô đến kết quả ghép cuối cùng.
> 4. Cách xử lý các trường hợp khó nhất (1 cửa, nhiều cửa giống hệt, hành lang đối xứng).
>
> Mục tiêu: một phương pháp đủ mạnh và mới để tiến tới bài báo hạng A*/Q1 — nhưng trước hết là **giải quyết được thực tế**.

---

## Bảng thuật ngữ nhanh (để đọc phần sau cho dễ)

| Từ dùng trong tài liệu | Nghĩa đơn giản |
|---|---|
| **Đám mây điểm** | Tập hàng triệu điểm 3D (x, y, z, màu) mô tả một không gian đã quét/tái dựng |
| **Phép ghép** | Phép biến đổi để đưa đám mây A "khớp" vào đám mây B = **xoay + dịch chuyển + phóng tỉ lệ đều** |
| **Cửa/Opening** | Cửa ra vào hoặc cửa sổ trên tường — dùng làm "mốc" để ghép |
| **Pháp tuyến** | Mũi tên vuông góc với một mặt phẳng (ví dụ mũi tên chĩa ra khỏi mặt tường) |
| **Trục "lên" (trọng lực)** | Hướng thẳng đứng từ sàn lên trần |
| **Camera / máy quay** | Vị trí và hướng của máy khi quay video — DA3 lưu lại được (`results.npz`) |
| **Suy biến / nhập nhằng** | Tình huống có **nhiều đáp án cùng đúng về mặt hình học**, không phân biệt được nếu chỉ nhìn cửa |

---

## PHẦN 1 — Bài toán là gì (giải thích bằng ví dụ)

Bạn có **hai đám mây điểm** dựng độc lập từ hai video khác nhau:
- **A** = phòng server (`h_server_room`)
- **B** = hành lang / phòng kế bên (`server_room`)

Hai không gian này **kề nhau và chung đúng MỘT bức tường**. Trên bức tường đó có **cửa** (mà cả hai video đều quay thấy).

Vì hai video quay riêng, mỗi đám mây nằm trong một hệ tọa độ **tùy ý**: xoay khác nhau, gốc khác nhau, và **tỉ lệ khác nhau** (đây là lý do hành lang bị "thu nhỏ một nửa" — thực chất là do tỉ lệ hai bản dựng lệch nhau, ví dụ 1.5 lần).

**Mục tiêu:** tìm **một phép ghép** (xoay + dịch + phóng tỉ lệ) đưa A áp đúng vào B, sao cho:
1. Bức tường chung của hai bên **trùng khít**.
2. Hai không gian nằm ở **hai phía đối diện** của tường (không đè chồng lên nhau).
3. Cả hai **cùng đứng thẳng** (sàn của A ngang với sàn của B, trần với trần).
4. Các cửa **ghép đúng cặp** (cửa nào khớp cửa nấy).

```
        Mong muốn:                          Đang bị lỗi:

   ┌─────────┐ │ ┌─────────┐          ┌─────────┐┌─────────┐
   │ PHÒNG A │ │ │ PHÒNG B │          │ PHÒNG A ││ PHÒNG B │  (chồng lên nhau)
   │  (sàn)  │[cửa] (sàn)  │          │ (TRẦN)  ││ (sàn)   │  (A bị lộn ngược:
   └─────────┘ │ └─────────┘          └─────────┘└─────────┘   sàn↔trần)
       tường chung                         tường chung
   Hai bên, thẳng đứng, sàn ngang sàn      Lộn ngược + sai phía
```

---

## PHẦN 2 — Vì sao khó: "Bức tường chung là một tấm gương đối xứng"

Đây là phần quan trọng nhất. **Chỉ nhìn vào các cửa trên một bức tường thì KHÔNG đủ để ghép đúng**, vì bức tường phẳng tạo ra một loạt **đáp án giả cùng khớp cửa hoàn hảo**. Có 4 kiểu nhập nhằng:

### Kiểu 1 — Lật trên/dưới (sàn ↔ trần)
Nếu ta lật đám mây A úp ngược lại, các cửa **vẫn nằm đúng chỗ trên tường** (vì cửa đối xứng trên–dưới khá tốt), nhưng sàn của A giờ nằm ở vị trí trần của B. → Đây chính là lỗi "**lộn ngược**" bạn thấy.

### Kiểu 2 — Sai phía (hai phòng đè lên nhau)
Cửa khớp nhưng cả hai phòng nằm **cùng một phía** của tường → chồng lấn. Đúng ra phải ở **hai phía**.

### Kiểu 3 — Xoay quanh trục tường (roll)
Xoay A quanh **đường thẳng nối hai tâm cửa**: hai cửa **giữ nguyên vị trí** với mọi góc xoay. Tức là có **vô số** góc xoay đều khớp cửa. Đây là nguồn gốc sâu xa nhất của nhập nhằng.

### Kiểu 4 — Hoán đổi cửa (swap)
Nếu hai cửa **giống hệt nhau** (cùng kích thước, hình dáng), thì "cửa 1 ↔ cửa 2" và "cửa 1 ↔ cửa 1" đều khớp như nhau → ghép nhầm cặp.

### Minh họa bằng CHÍNH dữ liệu của bạn

Tôi đã đo trên hai đám mây thật của bạn:

- **Phòng A** chỉ có **2 cửa**, và pháp tuyến hai cửa chỉ lệch nhau **0.7°** → **cùng một bức tường**.
- **Phòng B** có 6 cửa/cửa sổ trên **2 bức tường vuông góc**, nhưng 2 cái khớp với A lại nằm chung 1 tường.
- Tỉ lệ giữa hai bản dựng đo được = **1.509** (đây là lý do "thu nhỏ một nửa").

Khi ghép, máy tìm ra **4 đáp án đều khớp cửa y như nhau** (không cái nào "khớp hơn"):

| Đáp án | Ghép cặp cửa | Phía | Trên/dưới |
|---|---|---|---|
| 1 | (đúng cặp) | cùng phía | thẳng đứng |
| 2 | (đúng cặp) | **đối diện** | **lộn ngược** |
| 3 | (hoán cửa) | **đối diện** | **lộn ngược** |
| 4 | (hoán cửa) | cùng phía | thẳng đứng |

Nhìn bảng sẽ thấy nghiệt ngã: **"đối diện" luôn đi kèm "lộn ngược"**, còn "thẳng đứng" luôn đi kèm "cùng phía (chồng lấn)". **Không có đáp án nào vừa đối diện vừa thẳng đứng** — chỉ từ 2 cửa này.

### Vì sao lại như vậy? (chứng minh trực giác)
Hai tâm cửa nằm ngang, pháp tuyến tường cũng nằm ngang. Muốn đưa A sang **phía đối diện** của tường, ta phải **xoay 180°**. Nhưng vì cái trục để xoay 180° đó **nằm ngang**, nên xoay xong thì **sàn và trần bị đảo chỗ** → lộn ngược. Tức là "sang phía đối diện" và "lộn ngược" bị **khóa dính vào nhau**. Muốn tách chúng ra, ta cần **thông tin nằm ngoài cửa** (trọng lực, camera, hình dạng phòng...).

### Bảng: khi nào giải được, khi nào không

| Tình huống cửa | Kết quả |
|---|---|
| **1 cửa** | Rất khó — thiếu cả tỉ lệ lẫn hướng. Cần thông tin ngoài. |
| **2 cửa giống hệt, cùng 1 tường** (ca của bạn) | **Suy biến** — 4 đáp án như nhau. Không giải được nếu chỉ nhìn cửa. |
| **≥2 cửa khác nhau (cửa + cửa sổ), cùng 1 tường** | Hết nhầm cặp, nhưng **vẫn có thể lộn ngược** (vì vẫn 1 tường). |
| **≥3 cửa không thẳng hàng, cùng 1 tường** | Xác định được góc xoay, nhưng **còn nhập nhằng "lật gương"** qua tường. |
| **Cửa nằm trên ≥2 tường không song song** | **Giải được hoàn toàn** ✅ |

> **Kết luận Phần 2:** Lỗi bạn gặp **KHÔNG phải do thuật toán ghép sai**, mà do **đầu vào thiếu ràng buộc** (2 cửa giống hệt trên 1 tường). Về mặt toán học, không thuật toán nào chỉ nhìn cửa mà giải được. Đây chính là **khe hở khoa học** để làm điều mới.

---

## PHẦN 3 — Ý tưởng cốt lõi (bước ngoặt tư duy)

Thay vì cố **"tính ra một đáp án"** (bất khả thi khi suy biến), ta đổi cách làm:

> **BƯỚC 1 — Sinh ra một DANH SÁCH NHỎ các đáp án khả dĩ** (thường 2–8, đôi khi nhiều hơn). Tất cả đều khớp cửa hoàn hảo.
>
> **BƯỚC 2 — DÙNG BẰNG CHỨNG VẬT LÝ để LOẠI DẦN** cho đến khi còn 1 đáp án đúng.
>
> **BƯỚC 3 — Nếu bằng chứng không đủ để loại về 1, thì TRẢ VỀ danh sách còn lại một cách TRUNG THỰC** (kèm mức tin cậy), thay vì đoán bừa rồi ghép lộn ngược.

**Ví dụ đời thường:** Giống như bạn có 4 chìa khóa trông giống nhau và 1 ổ khóa. Bạn không "tính" ra chìa đúng — bạn **thử từng chìa** (dùng bằng chứng) để loại. Nếu 2 chìa đều mở được thì bạn thành thật nói "có 2 khả năng", chứ không nhắm mắt chọn đại.

Điểm hay: **chọn trong 4 đáp án** là bài toán **dễ hơn nhiều** so với "tính ra đáp án từ con số 0". Và đây đúng là chỗ mà các phương pháp ghép nối kinh điển (ICP, FGR, TEASER++) và cả học sâu (Predator, GeoTransformer) **bó tay**, vì chúng đòi hai đám mây phải **chồng nhau nhiều về thể tích** — còn ở đây chỉ chồng nhau đúng **một mặt tường**.

---

## PHẦN 4 — Các "bằng chứng" để chọn đáp án đúng (xếp từ MẠNH đến YẾU)

Mỗi bằng chứng riêng lẻ có thể yếu/nhiễu, nhưng **gộp lại thì đủ mạnh để quyết định**. Quan trọng: **bạn đã có sẵn dữ liệu cho những bằng chứng mạnh nhất** (camera).

### ⭐ Bằng chứng 1 — VỊ TRÍ CAMERA (mạnh nhất, và bạn CÓ SẴN)
Khi quay video phòng A, **máy quay luôn ở BÊN TRONG phòng A**. Tương tự, camera của B ở trong B.

→ Với mỗi đáp án ứng viên, kiểm tra: **camera của A có rơi về đúng một phía của tường, còn camera của B về phía kia (đối diện) không?** Đáp án đúng thì camera hai bên nằm **hai phía đối diện**; đáp án "cùng phía/chồng lấn" sẽ thấy camera hai bên lẫn vào nhau.

Đồng thời, **hướng "lên" của camera** (khi cầm máy quay bình thường) cho ta biết **trọng lực** → loại luôn đáp án lộn ngược.

DA3 của bạn **đã lưu vị trí + hướng camera** trong file `results.npz` của mỗi scene. **Đây là chìa khóa** — đừng vứt nó đi ở bước cắt cửa.

```
        Đáp án ĐÚNG:                       Đáp án SAI (chồng lấn):
   cam A ●    │    ● cam B            cam A ●  ●  ● cam B   (camera lẫn nhau
        ● ●   │   ● ●                      ● ● ● ●           → biết là sai phía)
      (trong A)│(trong B)
             tường
```

### ⭐ Bằng chứng 2 — NHÌN XUYÊN QUA CỬA ("cổng")
Coi mỗi cửa như một **ô cửa sổ nhìn sang không gian bên kia**. Nếu đứng trong phòng A nhìn qua cửa, ta thấy một phần của B (hoặc ít nhất là khung cửa, hèm cửa). Với đáp án đúng: **cái nhìn thấy qua cửa của A phải khớp với cái thực sự nằm ở đó trong B**.

Điều tuyệt vời: **hai cửa giống hệt nhau về hình dạng, nhưng "cảnh nhìn qua" chúng lại khác nhau** (vì phía sau mỗi cửa là chỗ khác nhau). → Bằng chứng này phá được cả 4 kiểu nhập nhằng cùng lúc (lật, phía, roll, hoán cửa).

Với **cửa đóng**: vẫn dùng được — hai video thấy **hai mặt đối diện** của cánh cửa (mặt trước vs mặt sau), bản thân điều đó cho biết phía.

### Bằng chứng 3 — TRỌNG LỰC / HƯỚNG LÊN
Hai không gian ở **cùng một tầng nhà** → sàn A và sàn B phải **cùng độ cao, cùng hướng**. Loại đáp án lộn ngược. Nguồn lấy trọng lực (mạnh → yếu): (a) hướng camera; (b) mặt sàn (mặt phẳng ngang lớn nhất, có điểm nằm phía trên); (c) người dùng bấm chọn 1 điểm trên sàn.
> Lưu ý: ước lượng trọng lực **tự động từ mỗi đám mây riêng lẻ hay bị sai** (như đã thấy). Nên ưu tiên lấy từ **camera**, hoặc **đối chiếu chéo giữa hai scene**.

### Bằng chứng 4 — KHÔNG GIAN TRỐNG (hai phòng không đè lên nhau)
Mỗi phòng chiếm một khối không gian riêng. Đáp án đúng: hai khối nằm **hai bên tường, không lồng vào nhau**. Đáp án sai thường làm khối A đâm xuyên vào khối B.

### Bằng chứng 5 — TƯỜNG CÓ ĐỘ DÀY (chi tiết dễ bị bỏ sót)
Hai video quét **hai mặt đối diện** của cùng bức tường, nên hai mặt đó **cách nhau đúng bằng độ dày tường** (vài cm đến vài chục cm), **không trùng khít tuyệt đối**. Nếu ép chúng trùng khít, ta có thể **phạt nhầm đáp án đúng**. → Mô hình hóa tường như một **tấm dày**, độ dày là ẩn số phụ.

### Bằng chứng 6 — NGỮ NGHĨA (yếu/phụ, dùng có kiểm soát)
Biển số phòng, chữ "EXIT", tay nắm/bản lề cửa, hướng chữ viết... có thể cho biết chiều. **Không nên** hỏi trực tiếp mô hình ngôn ngữ kiểu "đâu là trên" (dễ bịa, khó lặp lại). Nếu dùng, chỉ dùng dạng: cho xem 2–4 ảnh render ứng viên rồi hỏi **"ảnh nào trông hợp lý về vật lý"**, hoặc trích **chữ/biển số/segmentation sàn-tường** có thể kiểm chứng được.

### Bảng tổng kết bằng chứng nào phá được nhập nhằng nào

| Bằng chứng | Lật trên/dưới | Sai phía | Roll | Hoán cửa | Có sẵn trong data của bạn? |
|---|:---:|:---:|:---:|:---:|:---:|
| Vị trí camera | ✅ (qua hướng) | ✅✅ | ✅ | ~ | **CÓ** (`results.npz`) |
| Nhìn xuyên cửa | ✅ | ✅ | ✅ | ✅✅ | **CÓ** (ảnh + camera) |
| Trọng lực | ✅✅ | – | – | – | Một phần (từ camera) |
| Không gian trống | ~ | ✅✅ | ~ | ~ | CÓ (từ điểm) |
| Độ dày tường | – | ✅ | – | – | CÓ |
| Ngữ nghĩa | ✅ | ✅ | ✅ | ✅ | CÓ (ảnh) nhưng để sau |

---

## PHẦN 5 — KẾ HOẠCH CHI TIẾT TỪNG BƯỚC (pipeline hoàn chỉnh)

Đây là quy trình đầy đủ từ dữ liệu thô đến kết quả ghép cuối, kèm **đầu vào / đầu ra / cách làm / ví dụ** cho từng bước.

### Bước 0 — Chuẩn bị dữ liệu
- **Đầu vào:** với mỗi scene: đám mây điểm (x,y,z,màu) + **camera** (vị trí & hướng, từ `results.npz`) + (tùy chọn) các khung ảnh gốc.
- **Việc làm:** đọc `results.npz` lấy ma trận camera (extrinsics), gom về cùng hệ với đám mây. Cắt/segment các cửa (đang làm thủ công bằng CloudCompare — về sau tự động hóa).
- **Đầu ra:** mỗi scene = {đám mây, danh sách camera, danh sách cửa}.

### Bước 1 — Mô tả mỗi cửa KHÔNG phụ thuộc "hướng lên"
- **Cách làm:** với mỗi cụm điểm cửa, tính: **tâm**, **pháp tuyến** (dạng đường thẳng, không phân biệt chiều), **hai kích thước cạnh** (chữ nhật nhỏ nhất bao quanh), loại (cửa/cửa sổ). *Không* dùng "chiều rộng/cao" theo trục lên (vì trục lên hay sai).
- **Vì sao:** để mô tả cửa **ổn định** ngay cả khi chưa biết đâu là trên.
- *(Phần này code hiện tại `robust_align.py` đã làm.)*

### Bước 2 — SINH DANH SÁCH ỨNG VIÊN (đầy đủ theo đối xứng)
- **Cách làm:** ghép các cửa của A với B theo mọi cách **hợp lệ về hình học** (khớp khoảng cách + kích thước), rồi với mỗi cách nhân thêm các biến thể đối xứng: {chiều pháp tuyến} × {lật 180° trong mặt tường} × {đổi phía}. Tỉ lệ (scale) lấy từ **tỉ số khoảng cách giữa các cửa** (hoặc từ **kích thước một cửa** nếu chỉ có 1 cửa).
- **Đầu ra:** danh sách nhỏ các phép ghép ứng viên (ví dụ 4–8 cái cho ca của bạn). **Yêu cầu quan trọng:** danh sách phải **đầy đủ** (đáp án đúng chắc chắn nằm trong đó) — nếu bỏ sót đáp án đúng thì mọi bước sau vô nghĩa.
- **Lưu ý kỹ thuật (Codex nhấn mạnh):**
  - Với `n` cửa giống hệt, số cách ghép có thể tới `n!` — cần rút gọn bằng **nhóm đối xứng của cách bố trí cửa**, đừng liệt kê mù.
  - Cửa **vuông/tròn** → còn dư **một góc xoay liên tục** (không rời rạc) → phải xử lý riêng.
  - "Đổi phía" là **phép QUAY 180°** (hợp lệ), **không phải phép lật gương** — đừng tạo ra bản gương (đó là lỗi vật lý reviewer sẽ bắt).

### Bước 3 — LỌC bằng VỊ TRÍ CAMERA ⭐ (bước quyết định, làm sớm nhất)
- **Cách làm:** với mỗi ứng viên, đưa camera của A sang hệ của B. Kiểm tra:
  1. **Phía:** camera A và camera B có nằm **hai phía đối diện** của tường không? (dùng dấu của khoảng cách có hướng tới mặt tường).
  2. **Hướng lên:** hướng "lên" trung bình của camera A (sau phép ghép) có **cùng chiều** với của camera B không? → loại lộn ngược.
- **Đầu ra:** loại bỏ các ứng viên sai phía / lộn ngược. Dự kiến với ca của bạn: **từ 4 xuống còn 1**.
- **Ví dụ:** ứng viên "lộn ngược" sẽ có hướng-lên camera A ngược với B → loại ngay.

### Bước 4 — LỌC bằng NHÌN XUYÊN CỬA (portal) ⭐
- **Cách làm:** với mỗi cửa đã ghép, dựng "tia nhìn" từ camera A xuyên qua khung cửa. So sánh: cái mà A **thấy được qua cửa** (hình học + màu + đặc trưng ảnh) có khớp với cái **thực sự nằm ở đó trong B** không. Dùng **đặc trưng học sâu** (ví dụ đặc trưng ảnh nền tảng như DINO) thay vì so màu thô (vì ánh sáng/độ phơi sáng hai video khác nhau).
- **Đầu ra:** phá nốt nhập nhằng **hoán cửa** (hai cửa giống hệt nhưng "cảnh sau cửa" khác nhau).
- **Mẹo huấn luyện (nếu dùng mạng học):** mỗi cặp đúng **tự sinh ra** các bản sai (lật/đổi phía/hoán cửa) để làm **ví dụ phản chứng cực khó** — không cần gán nhãn thủ công.

### Bước 5 — CHẤM ĐIỂM bằng TRỌNG LỰC + KHÔNG GIAN TRỐNG (củng cố)
- **Cách làm:** thêm điểm phạt nếu (a) sàn A không ngang sàn B; (b) hai khối phòng đè lên nhau; (c) hai mặt tường không cách nhau một độ dày hợp lý. **Không** phạt vùng "chưa quét tới" (tránh phạt oan).
- **Đầu ra:** điểm tin cậy tổng hợp cho từng ứng viên.

### Bước 6 — CHỌN + HIỆU CHUẨN ĐỘ TIN CẬY (trả về "tập đáp án")
- **Cách làm:** gộp các bằng chứng thành xác suất cho từng ứng viên. Dùng kỹ thuật **hiệu chuẩn** (ví dụ conformal prediction) để quyết định:
  - Nếu **một** ứng viên vượt trội rõ → trả về **1 đáp án** (chắc chắn).
  - Nếu **hai** ứng viên sát nút và bằng chứng mâu thuẫn → trả về **cả hai + cảnh báo "nhập nhằng"**, kèm gợi ý đo thêm (xem Phần 6).
- **Vì sao quan trọng:** đây là điểm **trung thực** — thà nói "có 2 khả năng" còn hơn ghép lộn ngược mà tưởng đúng. Reviewer đánh giá cao sự trung thực có cơ sở.

### Bước 7 — TINH CHỈNH cục bộ (đánh bóng nghiệm đã chọn)
- **Cách làm:** với đáp án đã chọn, chạy tinh chỉnh (ví dụ ICP có màu) **chỉ trên vùng tường chung + camera**, và **chỉ chỉnh các hướng quan sát được** — **giữ nguyên lựa chọn đối xứng rời rạc** (không cho nó âm thầm lật lại).
- **Lưu ý (Codex):** ICP trên **một mặt phẳng** là bài toán suy biến → nếu chỉnh bừa sẽ trôi hoặc lật lại. Phải khóa các hướng không quan sát được.

### Bước 8 — ĐỒNG BỘ NHIỀU PHÒNG (nâng cấp mạnh nhất)
- **Ý tưởng:** nếu có **≥3 phòng nối thành vòng** (A–B–C–A), thì đi một vòng và nhân các phép ghép lại phải ra **phép đồng nhất** (về đúng chỗ cũ). Ràng buộc "đi vòng phải khép kín" này **tự loại** các đáp án lộn ngược ở từng cặp.
- **Sức mạnh:** **một cặp không thể giải riêng lẻ, lại giải được khi đặt trong vòng nhiều phòng.** Bạn đang có nhiều scene kề nhau (`connecting_space`, `hanh_lang_dai`, các server room, `thang_bo`, `thang_may`...) → đủ để làm thử một vòng 3–4 phòng. Đây có thể là **kết quả gây ấn tượng nhất** của cả nghiên cứu.

### Sơ đồ toàn bộ pipeline

```
[Đám mây + Camera + Cửa]
        │
   B1: Mô tả cửa (không cần "lên")
        │
   B2: Sinh DANH SÁCH ứng viên đầy đủ  ──►  ví dụ 4–8 phép ghép, đều khớp cửa
        │
   B3: LỌC bằng CAMERA (phía + hướng lên)   ⭐  ──► loại lộn ngược & sai phía
        │
   B4: LỌC bằng NHÌN XUYÊN CỬA (portal)     ⭐  ──► loại hoán cửa
        │
   B5: Chấm điểm trọng lực + không gian trống
        │
   B6: CHỌN + hiệu chuẩn  ──►  1 đáp án chắc chắn  HOẶC  tập đáp án + cảnh báo
        │
   B7: Tinh chỉnh cục bộ (giữ nhánh đối xứng)
        │
   B8: (nếu nhiều phòng) Đồng bộ theo vòng khép kín
        │
     KẾT QUẢ GHÉP
```

---

## PHẦN 6 — Xử lý các trường hợp KHÓ NHẤT

### Ca A: Chỉ có 1 cửa
- **Thiếu gì:** một cửa cho tâm + pháp tuyến + 2 cạnh chữ nhật (nếu cửa **không vuông** thì có hướng). Nhưng còn thiếu: chiều pháp tuyến (phía nào), lật trên/dưới.
- **Cần tối thiểu:** **1 hướng trọng lực có chiều** (từ camera) + **1 quan sát phía trong** (camera ở trong phòng nào) + **1 hướng cạnh cửa**. Có đủ 3 cái này (đều lấy được từ camera) → **giải được**.
- **Nếu cửa vuông/tròn:** còn dư một góc xoay → cần thêm **nhìn xuyên cửa** hoặc ngữ cảnh quanh cửa.

### Ca B: Nhiều cửa GIỐNG HỆT ở vị trí đối xứng (ca của bạn)
- **Trọng lực** giải được lật trên/dưới; **camera** giải được phía; nhưng **hoán cửa** thì trọng lực/camera-phía **không** giải được (vì đối xứng).
- **Chìa khóa:** **NHÌN XUYÊN CỬA** — mỗi cửa có "cảnh sau" khác nhau → phá hoán cửa. Hoặc: khoảng cách từ mỗi cửa tới **góc tường/mép tường** (ngữ cảnh bố trí) thường đã đủ.

### Ca C: Hành lang đối xứng HOÀN HẢO (thật sự bất khả thi)
Nếu mọi thứ đối xứng tuyệt đối (cửa giống nhau, cảnh hai bên giống nhau, camera đi đối xứng, không có trọng lực) thì **về mặt toán học không ai giải được** — hai đáp án cho **cùng một dữ liệu quan sát**.
- **Cách hành xử đúng:** **thừa nhận** và trả về cả hai đáp án + **"giấy chứng nhận bất khả thi"** (chứng minh hai đáp án không phân biệt được trong sai số đo).
- **Biến bế tắc thành hành động:** gợi ý **phép đo rẻ nhất** để gỡ: "bấm chọn 1 điểm trên sàn", "chỉ định 1 cặp cửa đúng", "chụp thêm 1 ảnh xuyên qua cửa X", hoặc "cắt thêm 1 cửa ở bức tường khác". → Đây cũng là một đóng góp hay (hệ thống biết **hỏi đúng câu** khi bí).

---

## PHẦN 7 — LỘ TRÌNH THỰC NGHIỆM (ra kết quả thật trước, viết báo sau)

**Giai đoạn 1 (làm ngay) — Bộ lọc CAMERA trên cặp thật của bạn.**
- Đọc camera từ `h_sever_room/results.npz` và `server_room/results.npz`.
- Chạy qua 4 ứng viên mà máy đã sinh, dùng "phía + hướng lên của camera" để chọn.
- **Kỳ vọng:** chọn ra đúng đáp án (đối diện + thẳng đứng + đúng cặp) — **giải quyết ngay ca đang lỗi**.
- Đây là **bằng chứng thực nghiệm đầu tiên** cho toàn bộ hướng đi.

**Giai đoạn 2 — Bộ lọc NHÌN XUYÊN CỬA.**
- Trích đặc trưng ảnh qua mỗi cửa; kiểm chứng khớp qua cổng. Xử lý được ca "nhiều cửa giống hệt".

**Giai đoạn 3 — Hiệu chuẩn "tập đáp án".**
- Cho hệ biết khi nào chắc chắn, khi nào phải trả về nhiều khả năng + hỏi thêm.

**Giai đoạn 4 — Đồng bộ nhiều phòng (vòng khép kín).**
- Ghép cả một dãy phòng của bạn; chứng minh nhập nhằng từng cặp bị vòng khép kín loại bỏ.

---

## PHẦN 8 — Vì sao đủ tầm bài báo A*/Q1 (tóm tắt để nhớ)

1. **Đặt lại bài toán:** ghép nối qua một-tường-chung là bài toán **"trả về một TẬP đáp án có đối xứng"**, không phải "một đáp án" — các phương pháp cũ ra một đáp án nên **giấu mất** sự nhập nhằng và ghép lộn ngược.
2. **Phương pháp:** *sinh ứng viên đầy đủ theo đối xứng* + *kiểm chứng bằng camera/nhìn-xuyên-cửa* → giải được chỗ mà **overlap thể tích bằng không** (chỉ chung một mặt phẳng) — nơi ICP/FGR/TEASER++/Predator/GeoTransformer đều thất bại.
3. **Trung thực có cơ sở:** trả về "tập đáp án có hiệu chuẩn" + "giấy chứng nhận bất khả thi" + "gợi ý đo thêm".
4. **Nâng cấp tòa nhà:** đồng bộ nhiều phòng theo vòng khép kín.
5. **Bộ dữ liệu:** video tự quay của bạn **vốn là các bản dựng độc lập** → không dính lỗi "cắt từ một mesh" (một điểm mạnh khi so với cách người khác tạo dữ liệu từ Matterport).

**So sánh (baseline) để chứng minh hơn:** ICP/point-to-plane, FGR, TEASER++, Predator, GeoTransformer, RoITr — chạy trên đúng bối cảnh suy biến này để cho thấy chúng lật/sai còn ta thì không.

**Thước đo:** tỉ lệ "đáp án đúng nằm trong danh sách", sai số xoay/dịch/tỉ lệ, **tỉ lệ lật ngược 180°**, độ chính xác "phía/trên-dưới/ghép cặp", và đường cong "độ phủ vs rủi ro" của tập đáp án.

---

## Việc tiếp theo tôi đề xuất bắt tay ngay
**Giai đoạn 1**: tôi dựng bộ lọc **camera-side** đọc `results.npz` của hai scene, chạy qua tập ứng viên `AMBIGUOUS` mà `robust_align.py` đã sinh, và chọn ra đáp án đúng. Nếu bạn đồng ý, tôi sẽ:
1. Kiểm tra nội dung `results.npz` (định dạng camera của DA3).
2. Viết hàm `verify_by_cameras(candidates, cams_A, cams_B, wall)` → trả điểm phía + hướng-lên cho từng ứng viên.
3. Chạy trên cặp `h_server_room` ↔ `server_room` và báo kết quả (kỳ vọng: hết lộn ngược, hết hoán cửa).
