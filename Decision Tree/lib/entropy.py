from typing import List, Any, Dict, TypeVar
from collections import Counter, defaultdict
import math

#hàm tính entropy binh thuong
def entropy(class_probabilities: List[float]) -> float:
    return sum(-p * math.log(p, 2) for p in class_probabilities if p > 0) # ignore zero probabilities

#hàm chủ yếu phục vụ cho việc tính entropy
#hàm tính sác xuất của các labels, hàm counter đếm số lần xuất hiện trong List
def class_probabilities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))

#hàm tính độ lợi 
def partition_entropy(subsets: List[List[Any]]) -> float:
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

#===============================          ===================================================


T = TypeVar('T') # generic type for inputs
# hàm trả về dict có key là giá trị của thuộc tính, value là List chứa các dòng có chứa thuộc tính đó
def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:

    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute) # value of the specified attribute
        partitions[key].append(input) # add input to the correct partition
    return partitions

#tính entropy của thuộc tính đó
def partition_entropy_by(inputs: List[Any], attribute: str, label_attribute: str) -> float:
    partitions = partition_by(inputs, attribute)
    labels = [[getattr(input, label_attribute) for input in partition] for partition in partitions.values()]
    return partition_entropy(labels)

