#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script kiem tra ham tien xu ly"""

import sys
import os
os.chdir('D:\\abstractive_summarization_vit5')
sys.path.insert(0, 'scripts')

# Set encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from preprocess import clean_text

# Cac mau test tu file train_raw_sample.txt
test_samples = {
    "Mau 1 - Tac gia/Nguon": '''Nếu đã là fan của vũ trụ Marvel, ắt hẳn bạn sẽ không còn xa lạ gì với Captain Marvel - nữ siêu anh hùng mới nhất. "Captain Marvel" dự kiến khởi chiếu vào ngày 8/3/2019./.
Hà Phương/VOV.VN.''',

    "Mau 2 - Thuc hien": '''Các người đẹp luôn phải nỗ lực hết mình trong những hoạt động của cuộc thi. Trước khi bước vào đêm chung kết, tối 5.12, 38 người đẹp sẽ tham gia một sự kiện lớn của chương trình.
Thực hiện: Hiền Nhi - Độc Lập.''',

    "Mau 3 - Khoang trang kep": '''Theo đó, với xuất thân là một người lính  , Carol Denvers chưa bao giờquên những ngày tháng phục vụ. Nhưng một mặt khác, cô chính là Captain Marvel  cái tên đã nói lên tất cả.''',

    "Mau 4 - Bai viet gop": '''Mendes đã có mặt ở Manchester.
Siêu cò phản bội M.U vụ Otamendi. Ban đầu, nhà môi giới Jorge Mendes hứa sẽ thuyết phục thân chủ của mình là Nicolas Otamendi gia nhập M.U.
Abramovich bật đèn xanh, Chelsea kích nổ 3 bom tấn.''',

    "Mau 5 - Tac gia phuc tap": '''Trong khi đó, chỉ số HNX - Index trên sàn Hà Nội chỉ tăng trong ít phút đầu giao dịch. Chỉ số này kết phiên ở 101,57 điểm, giảm 0,5 điểm.
(Lê Thịnh). Trường Giang.''',

    "Mau 6 - Ngay thang": '''Sự kiện này diễn ra vào ngày 12-10-2023 hoặc 15/11/2023 hoặc 20.09.2023. Cung cấp thêm thông tin từ ngày 28.11 cho đến 5.12 năm ngoái.''',
}

print("=" * 80)
print("KIEM TRA HAM TIEN XU LY CLEAN_TEXT")
print("=" * 80)

for title, text in test_samples.items():
    print(f"\n{title}")
    print("-" * 80)
    print(f"INPUT:\n{text}")
    print(f"\nOUTPUT:\n{clean_text(text)}")
    print()
