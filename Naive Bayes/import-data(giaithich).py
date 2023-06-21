from io import BytesIO # So we can treat bytes as a file.
import requests
import tarfile
# File data là định dạng .tar.bz


# Chọn đường dẫn tới nơi chứa data 
BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"

# Tên các file tải về và giải nén, được sử dụng làm data 
FILES = ["20021010_easy_ham.tar.bz2",
"20021010_hard_ham.tar.bz2",
"20021010_spam.tar.bz2"]

# Tên thư mục lưu trữ các file data tải về
OUTPUT_DIR = 'spam_data'

for filename in FILES:
    # Gán nội dung tải về vào biến content bằng cách dùng requests.get() để tải nội dung từ URL của mỗi tệp dữ liệu.
    content = requests.get(f"{BASE_URL}/{filename}").content

    # Dùng BytesIO để có thể sử dụng content như 1 tệp tin (file)
    fin = BytesIO(content)

    # Mở file nén có định dạng .tar.bz2
    with tarfile.open(fileobj=fin, mode='r:bz2') as tf:

        # Giải nén tất cả các file và lưu file trong OUTPUT_DIR
        tf.extractall(OUTPUT_DIR)
