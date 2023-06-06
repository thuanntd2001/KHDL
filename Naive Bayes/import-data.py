from io import BytesIO # So we can treat bytes as a file.
import requests
# To download the files, which
import tarfile
# are in .tar.bz format.
BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
"20021010_hard_ham.tar.bz2",
"20021010_spam.tar.bz2"]
# This is where the data will end up,
# in /spam, /easy_ham, and /hard_ham subdirectories.
# Change this to where you want the data.
OUTPUT_DIR = 'spam_data'
for filename in FILES:
    # Use requests to get the file contents at each URL.
    content = requests.get(f"{BASE_URL}/{filename}").content
    # Wrap the in-memory bytes so we can use them as a "file."
    fin = BytesIO(content)
    # And extract all the files to the specified output dir.
    with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
        tf.extractall(OUTPUT_DIR)
