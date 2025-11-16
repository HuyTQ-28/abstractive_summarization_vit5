import logging
import pandas as pd
import re
import unicodedata
from tqdm.auto import tqdm
import argparse
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from pathlib import Path

tqdm.pandas()

# Cấu hình
DATASET_NAME = "vietgpt/news_summarization_vi"
PROCESSED_DATA_DIR = "data/processed"
MIN_ARTICLE_LEN = 100
MIN_SUMMARY_LEN = 15


def normalize_dates(text):
    # Thay thế các dấu - . trong ngày tháng thành /
    text = re.sub(r'(\d{1,2})[-.\s](\d{1,2})[-.\s](\d{2,4})', r'\1/\2/\3', text)
    # Xử lý trường hợp chỉ có ngày và tháng
    text = re.sub(r'(\d{1,2})[-.\s](\d{1,2})(?![\d/])', r'\1/\2', text)
    return text


def remove_metadata_lines(text: str) -> str:
    """
    Loại bỏ các dòng meta như chữ ký tác giả, nguồn tin... thường xuất hiện
    ở cuối bài báo (ví dụ: "Hà Phương/VOV.VN.", "Thực hiện: Hiền Nhi - Độc Lập.").
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Bỏ dòng trống
            continue

        # Các dòng bắt đầu bằng cụm từ meta phổ biến
        if re.match(r'^(thực hiện|tác giả|biên soạn|dịch giả)\b', stripped, flags=re.IGNORECASE):
            continue

        # Các dòng chứa tên tòa soạn/báo điện tử dạng "Tên tác giả/VOV.VN"
        if re.search(r'\/vov\.vn|\/vnexpress\.net|\/thanhnienonline|\/thanhnien\.vn|\/tuoitre\.vn', stripped,
                     flags=re.IGNORECASE):
            continue

        # Các dòng ghi dạng "Tên tác giả/TÊN BÁO."
        if re.match(r'^[A-ZÀ-Ỵ][^,]+\/[A-Z0-9.]+\.?$', stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def clean_text(text):
    """Hàm xử lý văn bản"""

    # Chuẩn hóa Unicode và chuyển văn bản thành chữ thường
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()

    # Loại bỏ các dòng meta (tác giả, nguồn tin...) trước
    text = remove_metadata_lines(text)

    # Loại bỏ nhiễu (htmls, urls và kí tự rỗng, chú thích ảnh/video)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '', text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

    # Xóa chú thích ảnh/video
    text = re.sub(r'\s*(ảnh|video)\s*:\s*[^\n\r]+', ' ', text)

    # Loại bỏ kí tự đặc biệt
    text = re.sub(r'[^\w\s,.?!:;()%/$đ-]', '', text)
    text = re.sub(r'\((nguồn|theo).*?\)\s*', '', text)

    # Thêm khoảng trắng quanh dấu câu (trừ dấu / để không làm đổi định dạng ngày và đơn vị)
    text = re.sub(r'\s*([?!:;()\\])\s*', r' \1 ', text)
    text = re.sub(r'((?<!\d)[.,]|[.,](?!\d))', r' \1 ', text)

    # Gộp các dấu câu lặp lại (ví dụ: "..." -> ".")
    text = re.sub(r'([.?!,])(\s*\1)+', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Hàm xử lý dữ liệu"""
    # 1. Loại bỏ dòng bị thiếu và trùng lặp
    initial_rows = len(df)
    df.dropna(subset=['content', 'summary'], inplace=True)
    df.drop_duplicates(subset=['content', 'summary'], inplace=True, keep='first')
    filtered_rows = len(df)
    logging.info(f'Đã loại bỏ {initial_rows - filtered_rows} dòng bị thiếu và trùng lặp')

    # 2. Chuẩn hóa và làm sạch văn bản
    df['content'] = df['content'].progress_apply(clean_text)
    df['summary'] = df['summary'].progress_apply(clean_text)

    # 3. Lọc dữ liệu văn bản theo độ dài
    df['content_len'] = df['content'].apply(lambda x: len(x.split()))
    df['summary_len'] = df['summary'].apply(lambda x: len(x.split()))

    # Áp dụng các bộ lọc theo độ dài
    df = df[df['content_len'] >= MIN_ARTICLE_LEN]
    df = df[df['summary_len'] >= MIN_SUMMARY_LEN]

    # Loại bỏ cột thừa
    df.drop(columns=['content_len', 'summary_len'], inplace=True)
    logging.info(f'Đã loại bỏ {filtered_rows - len(df)} dòng văn bản không phù hợp')
    logging.info(f'Kích thước cuối cùng của dataset: {len(df)} dòng văn bản phù hợp')

    return df

def main(input_dir: str, output_dir: str, split_name: str):
    logging.info(f'Bắt đầu xử lý dữ liệu từ tập {split_name}')
    df = pd.read_csv(f'{input_dir}/{split_name}.csv')
    df = preprocess_data(df)
    df.to_csv(f'{output_dir}/{split_name}.csv', index=False)
    logging.info(f'Đã xử lý dữ liệu từ tập {split_name} và lưu vào {output_dir}/{split_name}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline tiền xử lý dữ liệu cho dự án tóm tắt văn bản.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/raw",
        help="Thư mục chứa các file CSV gốc."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Thư mục để lưu các file CSV đã xử lý."
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Tên tập dữ liệu."
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.split_name)