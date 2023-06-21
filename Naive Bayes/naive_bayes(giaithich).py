from typing import Set
import re

def tokenize(text: str) -> Set[str]:

     # Chuyển tất cả nội dung văn bản thành chữ thường 
    text = text.lower()  

    # Tìm kiếm các từ có trong văn bản [a-z0-9']: a -> z, 0 -> 9, dấu nháy đơn                   
    all_words = re.findall("[a-z0-9']+", text)  

    # Loại bỏ các từ trùng lập 
    return set(all_words)                 S  

# assert dùng để đảm bảo hàm tokenize chạy đúng với kết quả định sẵn, sai thì sẽ ngừng và thông báo AssertionError 
assert tokenize("Data Science is science") == {"data", "science", "is"}



from typing import NamedTuple

# Định dạng Message thành tuple 
class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        # Tập hợp lưu trữ các từ xuất hiện tin nhắn  
        self.tokens: Set[str] = set()

        # Tập hợp lưu trữ các từ xuất hiện trong tin nhắn spam  
        self.token_spam_counts: Dict[str, int] = defaultdict(int)

        # Tập hợp lưu trữ các từ xuất hiện không phải là tin nhắn spam  
        self.token_ham_counts: Dict[str, int] = defaultdict(int)

        # Khởi tạo số lượng tin nhắn spam / không spam với giá trị ban đầu = 0 
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # Nếu là tin nhắn spam thì tin nhắn spam +1 còn không phải thì tin nhắn không spam +1 
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # Tách từ trong message.text bằng tokenize 
            for token in tokenize(message.text):
                self.tokens.add(token)

                # Nếu là tin nhắn spam thì từ trong tin nhắn spam +1 và ngược lại 
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:

        # Lấy số lần xuất hiện của từ trong các tin nhắn spam/ham từ từ điển token_spam_counts lưu vào biến spam/ham. 
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        # Tính xác xuất = công thức với: 
        # spam: số lần xuất hiện của các từ trong tin nhắn spam
        # self.k: smoothing factor 
        # self.spam_messages: tổng số tin nhắn spam.
        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        # Trả về xác suất của từ trong tin nhắn spam và tin nhắn không phải spam.
        return p_token_spam, p_token_ham

    dSef predict(self, text: str) -> float:

        # Chuyển text thành các từ 
        text_tokens = tokenize(text)

        log_prob_if_spam = log_prob_if_ham = 0.0

        for token in self.tokens:
            # Tính xác xuất = _probabilities
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # Nếu từ xuất hiện thì thêm vào trong xác xuất xuất hiện của spam/ham 
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # Ngược lại, nếu từ không xuất hiện thì thêm vào xác xuất không xuất hiện 
            # log(1 - xác xuất xuất hiện)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        # Tính hàm mũ 
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)

        # Trả về giá trị xác xuất 
        # Giá trị từ 0 -> 1 
        return prob_if_spam / (prob_if_spam + prob_if_ham)

# Tạo các message có nội dung và trạng thái tương ứng 
messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

# Model sử dụng class NaiveBayesClassifier và train 
model = NaiveBayesClassifier(k=0.5)
model.train(messages)
s
# Đảm bảo mọi thứ hoạt động đúng bằng hàm assert 
assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

# Đánh giá xác suất dự đoán cho "hello spam" 
text = "hello spam"

# Smoothing Laplace 
probs_if_spam = [
    # Đây là xác suất điều kiện của từ "spam" trong text nếu nó có xuất hiện
    # Ta tính số lần từ "spam" xuất hiện trong văn bản spam cộng thêm với một giá trị k,
    # sau đó chia cho tổng số văn bản spam (1) cộng với 2 lần giá trị k (2 * 0.5).
    # Điều này giúp xác định xác suất từ "spam" trong văn bản text nếu nó là spam.
    (1 + 0.5) / (1 + 2 * 0.5),      # "spam"

    # Đây là xác suất điều kiện của từ "ham" trong text nếu nó không xuất hiện
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)

    # Đây là xác suất điều kiện của từ "rules" trong text nếu nó không xuất hiện
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)

    # Đây là xác suất điều kiện của từ "hello" trong text nếu nó có xuất hiện.
    (0 + 0.5) / (1 + 2 * 0.5)       # "hello" (present)s
]

probs_if_ham = [
    
    # Đây là xác suất điều kiện của từ "spam" trong text nếu nó có xuất hiện
    # ta tính số lần từ "spam" xuất hiện trong văn bản không spam cộng thêm với một giá trị k 
    # sau đó chia cho tổng số văn bản không spam (2) cộng với 2 lần giá trị k (2 * 0.5).
    # Điều này giúp xác định xác suất từ "spam" trong văn bản text nếu nó không phải là spam.
    (0 + 0.5) / (2 + 2 * 0.5),      # "spam"  (present)

    # Đây là xác suất điều kiện của từ "ham" trong text nếu nó không xuất hiện
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)

    # Đây là xác suất điều kiện của từ "rules" trong text nếu nó không xuất hiện
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)

    # : Đây là xác suất điều kiện của từ "hello" trong văn bản text nếu nó có xuất hiện
    (1 + 0.5) / (2 + 2 * 0.5),      # "hello" (present)
]

# Tính xác xuất bằng công thức Bayes và logarit 
p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

# Đảm bảo mọi thứ hoạt động đúng 
assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)

# Xóa ký tự "s" cuối cùng của từ word. Nếu word kết thúc bằng "s", chúng ta sẽ thay thế "s" đó bằng một chuỗi rỗng, tức là xóa bỏ ký tự "s".
def drop_final_s(word):
    return re.sub("s$", "", word)

def main():
    import glob, re
    
    # Đường dẫn để file 
    path = 'spam_data/*/*'
    
    data: List[Message] = []
    
    # Hàm glob.glob trả về danh sách các tên tệp tin khớp với mẫu
    for filename in glob.glob(path):
        is_spam = "ham" not in filename
    
        # Có một số ký tự rác trong các email, tham số errors='ignore'
        # sẽ bỏ qua thay vì ném ra exception.
        with open(filename, errors='ignore') as email_file:
            for line in email_file:
                if line.startswith("Subject:"):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(subject, is_spam))
                    break  
    
    import random
    from  lib.machine_learning import split_data
    
    # Tạo ngẫu nhiên seed cho việc chia dữ liệu huấn luyện và kiểm tra.
    random.seed(0)  

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
  
    train_messages, test_messages = split_data(data, 0.75)
    
    # Huấn luyện mô hình NaiveBayesClassifier trên tập huấn luyện.
    model = NaiveBayesClassifier()
    model.train(train_messages)
    
    from collections import Counter

    # Dự đoán nhãn và tính ma trận nhầm lẫn trên tập kiểm tra.
    predictions = [(message, model.predict(message.text))
                   for message in test_messages]
    
    # Giả định rằng xác suất spam > 0.5 tương ứng với dự đoán spam
    # và đếm số lượng các kết hợp (is_spam thực tế, is_spam dự đoán)
    confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                               for message, spam_probability in predictions)
    
    print(confusion_matrix)
    
    # Định nghĩa hàm p_spam_given_token để tính xác suất spam cho từng từ sử dụng phương pháp _probabilities.
    def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
      
        prob_if_spam, prob_if_ham = model._probabilities(token)
    
        return prob_if_spam / (prob_if_spam + prob_if_ham)
    
    # Sắp xếp các từ theo thứ tự xác suất spam tăng dần và giảm dần.
    words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))
    
    # In ra danh sách các từ có xác suất spam cao nhất và không phải spam cao nhất.
    print("spammiest_words", words[-10:])
    print("hammiest_words", words[:10])
    
if __name__ == "__main__": main()
