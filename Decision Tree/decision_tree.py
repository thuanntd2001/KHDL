from typing import NamedTuple, Union, Any
from lib.entropy import *
from lib.data import *

class Leaf(NamedTuple):
    value: Any
class Split(NamedTuple):
    attribute: str
    subtrees: dict
    default_value: Any = None
DecisionTree = Union[Leaf, Split]


#hàm phân loại sau khi có cây
def classify(tree: DecisionTree, input: Any) -> Any:
    if isinstance(tree, Leaf):
        return tree.value
    subtree_key = getattr(input, tree.attribute)
    if subtree_key not in tree.subtrees: 
        return tree.default_value 
    subtree = tree.subtrees[subtree_key] 
    return classify(subtree, input) 

#Xây cây
def build_tree_id3(inputs: List[Any], split_attributes: List[str], 
                   target_attribute: str) -> DecisionTree:

    label_counts = Counter(getattr(input, target_attribute) for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]

    if len(label_counts) == 1:
        return Leaf(most_common_label)

    if not split_attributes:
        return Leaf(most_common_label)

    def split_entropy(attribute: str) -> float:
        return partition_entropy_by(inputs, attribute, target_attribute)
    best_attribute = min(split_attributes, key=split_entropy)
    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]

    subtrees = {attribute_value : build_tree_id3(subset, new_attributes, target_attribute) 
                for attribute_value, subset in partitions.items()}
    return Split(best_attribute, subtrees, default_value=most_common_label)


tree = build_tree_id3(inputs, ['level', 'lang', 'tweets', 'phd'], 'did_well')
# Should predict True
print(classify(tree, Candidate("Junior", "Java", True, False)))
# Should predict False
print(classify(tree, Candidate("Junior", "Java", True, True)))

print(classify(tree, Candidate("Intern", "Java", True, True)))
