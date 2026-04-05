import xml.etree.ElementTree as ET
import os
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


def write_preprocessed_tree(tree: ET.ElementTree, output_path: str, indent: str = "    ") -> None:
    ET.indent(tree, space=indent, level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True, method="xml")


def convert_semeval14_to_15_16_format(semeval14_tree: ET.ElementTree) -> ET.ElementTree:
    """Build SemEval15/16-style Reviews XML from a SemEval14 root. Returns a new tree (does not write)."""
    root: ET.Element = semeval14_tree.getroot()

    new_root: ET.Element = ET.Element("Reviews")
    review: ET.Element = ET.SubElement(new_root, "Review")
    sentences: ET.Element = ET.SubElement(review, "sentences")

    for sentence in root.findall("sentence"):
        sentence_id = sentence.get("id")
        text = sentence.find("text").text

        new_sentence = ET.SubElement(sentences, "sentence", id=sentence_id)
        ET.SubElement(new_sentence, "text").text = text

        opinions = ET.SubElement(new_sentence, "Opinions")

        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is not None:
            for aspect in aspect_terms.findall("aspectTerm"):
                target = aspect.get("term")
                polarity = aspect.get("polarity")
                from_idx = aspect.get("from")
                to_idx = aspect.get("to")
                ET.SubElement(opinions, "Opinion", target=target, category="", polarity=polarity, from_=from_idx, to=to_idx)
        else:
            ET.SubElement(opinions, "Opinion", target="NULL")

    return ET.ElementTree(new_root)

# Definition to delete the implict aspect from the dataset (only works on 2015/2016-style Reviews XML)
def delete_implicit_aspects_and_conflicting_polarities(tree: ET.ElementTree) -> None:
    """
    Deletes the implicit aspects (target = null) from the dataset.
    Deletes any opinions with a conflicting polarity (polarity = conflict).
    Deletes any sentences without any non-null opinions.
    Mutates tree in place.
    """
    root: ET.Element = tree.getroot()

    amount_null_target_opinions: int = 0
    amount_conflicting_polarities: int = 0
    amount_sentences_removed: int = 0

    for review in root.findall('.//Review'):
        sentences: ET.Element = review.find('sentences')
        if sentences is None:
            continue

        sentences_to_remove: list[ET.Element] = []

        for sentence in sentences.findall('sentence'):
            opinions_elem = sentence.find('Opinions')
            if opinions_elem is None:
                sentences_to_remove.append(sentence)
                continue

            for opinion in list(opinions_elem.findall('Opinion')):
                if opinion.get("target") == "NULL":
                    opinions_elem.remove(opinion)
                    amount_null_target_opinions += 1

            for opinion in list(opinions_elem.findall('Opinion')):
                if opinion.get("polarity") == "conflict":
                    opinions_elem.remove(opinion)
                    amount_conflicting_polarities += 1

            if not opinions_elem.findall("Opinion"):
                sentences_to_remove.append(sentence)

        for sentence in sentences_to_remove:
            sentences.remove(sentence)

        amount_sentences_removed += len(sentences_to_remove)

    print("delete_implicit_aspects_and_conflicting_polarities:")
    print(f"  null target opinions removed: {amount_null_target_opinions}")
    print(f"  conflict polarities removed: {amount_conflicting_polarities}")
    print(f"  sentences removed (no opinions left): {amount_sentences_removed}")

def remove_intersections(training_tree: ET.ElementTree, test_tree: ET.ElementTree) -> None:
    """
    Removes any sentences from the training tree that also appear in the test tree (by sentence text).
    Mutates training_tree in place.
    """

    def text_from_sentence(sentence: ET.Element) -> str:
        return sentence.find('text').text.strip()

    test_root: ET.Element = test_tree.getroot()
    test_sentences: set[str] = {text_from_sentence(sentence) for sentence in test_root.findall('.//sentence')}

    training_root: ET.Element = training_tree.getroot()

    amount_sentences_removed: int = 0

    for review in training_root.findall('.//Review'):
        sentences = review.find('sentences')
        if sentences is None:
            continue

        sentences_to_remove: list[ET.Element] = []
        for sentence in sentences.findall('sentence'):
            if text_from_sentence(sentence) in test_sentences:
                sentences_to_remove.append(sentence)

        for sentence in sentences_to_remove:
            sentences.remove(sentence)

        amount_sentences_removed += len(sentences_to_remove)

    print("remove_intersections:")
    print(f"  training sentences removed (overlap with test): {amount_sentences_removed}")

def remove_duplicate_opinions(tree: ET.ElementTree) -> None:
    """
    Per sentence: group opinions by (target, category).

    - Same (target, category) with different polarities -> remove every opinion in that group.
    - Same (target, category) with the same polarity (duplicates) -> keep one, remove the rest.

    Mutates tree in place.
    """
    root: ET.Element = tree.getroot()

    removed_conflicting_groups: int = 0
    removed_duplicate_same_polarity: int = 0

    for review in root.findall('.//Review'):
        sentences = review.find('sentences')
        if sentences is None:
            continue

        for sentence in sentences.findall('sentence'):
            opinions_elem = sentence.find('Opinions')
            if opinions_elem is None:
                continue

            opinions = list(opinions_elem.findall('Opinion'))
            if len(opinions) <= 1:
                continue

            groups: dict[tuple[str, str], list[ET.Element]] = defaultdict(list)
            for op in opinions:
                target = op.get('target') or ''
                category = op.get('category') or ''
                groups[(target, category)].append(op)

            opinions_to_remove: list[ET.Element] = []
            for _key, group_ops in groups.items():
                if len(group_ops) <= 1:
                    continue
                polarities = {op.get('polarity') or '' for op in group_ops}
                if len(polarities) > 1:
                    opinions_to_remove.extend(group_ops)
                    removed_conflicting_groups += 1
                else:
                    opinions_to_remove.extend(group_ops[1:])
                    removed_duplicate_same_polarity += len(group_ops) - 1

            for opinion in opinions_to_remove:
                opinions_elem.remove(opinion)

    print("remove_duplicate_opinions:")
    print(f"  conflicting (target,category) groups removed entirely: {removed_conflicting_groups}")
    print(f"  redundant duplicate opinions removed (same polarity): {removed_duplicate_same_polarity}")

# (dataset_id, raw_xml_path, preprocessed_output_path, convert_from_semeval14)
_DATASET_SPECS: list[tuple[str, str, str, bool]] = [
    ("s14_rest_train", os.getenv("PATH_TO_RAW_SEMEVAL_14_RESTAURANTS_TRAIN_DATA"), os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_14_RESTAURANTS_TRAIN_DATA"), True),
    ("s14_rest_test", os.getenv("PATH_TO_RAW_SEMEVAL_14_RESTAURANTS_TEST_DATA"), os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_14_RESTAURANTS_TEST_DATA"), True),
    ("s14_lap_train", os.getenv("PATH_TO_RAW_SEMEVAL_14_LAPTOPS_TRAIN_DATA"), os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_14_LAPTOPS_TRAIN_DATA"), True),
    ("s14_lap_test", os.getenv("PATH_TO_RAW_SEMEVAL_14_LAPTOPS_TEST_DATA"), os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_14_LAPTOPS_TEST_DATA"), True),
    ("s15_rest_train", os.getenv("PATH_TO_RAW_SEMEVAL_15_RESTAURANTS_TRAIN_DATA"), os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TRAIN_DATA"), False),
    ("s15_rest_test", os.getenv("PATH_TO_RAW_SEMEVAL_15_RESTAURANTS_TEST_DATA"), os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_15_RESTAURANTS_TEST_DATA"), False),
    ("s16_rest_train", os.getenv("PATH_TO_RAW_SEMEVAL_16_RESTAURANTS_TRAIN_DATA"), os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_16_RESTAURANTS_TRAIN_DATA"), False),
    ("s16_rest_test", os.getenv("PATH_TO_RAW_SEMEVAL_16_RESTAURANTS_TEST_DATA"), os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_16_RESTAURANTS_TEST_DATA"), False),
]

_TRAIN_TEST_INTERSECTION_PAIRS: list[tuple[str, str]] = [
    ("s14_rest_train", "s14_rest_test"),
    ("s15_rest_train", "s15_rest_test"),
    ("s16_rest_train", "s16_rest_test"),
    ("s14_lap_train", "s14_lap_test"),
]

trees: dict[str, ET.ElementTree] = {}
for dataset_id, raw_path, _out_path, from_semeval14 in _DATASET_SPECS:
    raw_tree = ET.parse(raw_path)
    trees[dataset_id] = convert_semeval14_to_15_16_format(raw_tree) if from_semeval14 else raw_tree

for dataset_id in trees:
    print(f"\n=== {dataset_id} — delete_implicit_aspects_and_conflicting_polarities ===")
    delete_implicit_aspects_and_conflicting_polarities(trees[dataset_id])

for train_id, test_id in _TRAIN_TEST_INTERSECTION_PAIRS:
    print(f"\n=== intersection: {train_id} vs {test_id} ===")
    remove_intersections(trees[train_id], trees[test_id])

for dataset_id, _raw_path, output_path, _from14 in _DATASET_SPECS:
    print(f"\n=== {dataset_id} — remove_duplicate_opinions & save ===")
    # remove_duplicate_opinions(trees[dataset_id])
    write_preprocessed_tree(trees[dataset_id], output_path)
    print(f"Saved: {output_path}")