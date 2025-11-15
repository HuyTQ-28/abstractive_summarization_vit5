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


def normalize_numbers(text):
    text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
    while re.search(r'(\d+)\.(\d{3})', text):
        text = re.sub(r'(\d+)\.(\d{3})', r'\1\2', text)
    return text

def normalize_dates(text):
    # Thay thế các dấu - . trong ngày tháng thành /
    text = re.sub(r'(\d{1,2})[-.\s](\d{1,2})[-.\s](\d{2,4})', r'\1/\2/\3', text)
    # Xử lý trường hợp chỉ có ngày và tháng
    text = re.sub(r'(\d{1,2})[-.\s](\d{1,2})(?![\d/])', r'\1/\2', text)
    return text

def clean_text(text):
    """Hàm xử lý văn bản"""
    # Chuẩn hóa Unicode và chuyển văn bản thành chữ thường
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()

    # Loại bỏ nhiễu (htmls, urls và kí tự rỗng, chú thích ảnh/video)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '', text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    text = re.sub(r'\s*(ảnh|video)\s*:\s*[\w\s_.-]+', ' ', text)

    # Loại bỏ kí tự đặc biệt
    text = re.sub(r'[^\w\s,.?!:;()/\\-]', '', text)
    text = re.sub(r'\((nguồn|theo).*?\)\s*', '', text)

    # Các bước chuẩn hóa
    text = normalize_dates(text)
    text = normalize_numbers(text)

    # Thêm khoảng trắng quanh dấu câu, xử lý đặc biệt cho .,/
    text = re.sub(r'\s*([?!:;()\\])\s*', r' \1 ', text)
    text = re.sub(r'((?<!\d)[.,]|[.,](?!\d))', r' \1 ', text)
    text = re.sub(r'(?<!\d)\s*/\s*|\s*/\s*(?!\d)', r' / ', text)

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

def main(output_dir: str):
    logging.info(f'Bắt đầu xử lý dữ liệu từ dataset {DATASET_NAME}')
    dataset = load_dataset(DATASET_NAME)
    df = pd.concat([dataset[split].to_pandas() for split in dataset.keys()])
    df = preprocess_data(df)
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=(0.1/0.9), random_state=42)

    logging.info(f'Kích thước của train set: {len(train_df)}')
    logging.info(f'Kích thước của val set: {len(val_df)}')
    logging.info(f'Kích thước của test set: {len(test_df)}')

    # Lưu kết quả
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_path / 'train.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv(output_path / 'val.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv(output_path / 'test.csv', index=False, encoding='utf-8-sig')
    logging.info(f'Đã lưu kết quả vào {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline tiền xử lý dữ liệu cho dự án tóm tắt văn bản.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Thư mục để lưu các file CSV đã xử lý."
    )
    args = parser.parse_args()
    main(args.output_dir)