import xml.etree.ElementTree as ET
import os
from enum import StrEnum
from dotenv import load_dotenv

class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class DataSet:

    def __init__(self, file_path: str):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")
        if not file_path.endswith(".xml"):
            raise ValueError(f"The file {file_path} is not an XML file")

        self.file_path = file_path
        self.tree: ET.ElementTree = ET.parse(self.file_path)

    @property
    def root(self) -> ET.Element:
        return self.tree.getroot()

    @property
    def all_sentences_as_text(self) -> list[str]:
        return [sentence.find('text').text for sentence in self.root.findall('.//sentence')]

    @property
    def all_sentences_with_aspects_and_polarities(self) -> list[tuple[str, list[tuple[str, Polarity]]]]:
        return [
            (
                sentence.find('text').text, 
                [(aspect.get('target'), Polarity(aspect.get('polarity'))) for aspect in sentence.find('Opinions').findall('Opinion')]
            ) for sentence in self.root.findall('.//sentence')
        ]

    def polarity_frequencies(self) -> dict:

        polarity_counts: dict[Polarity, int] = {polarity: 0 for polarity in Polarity}

        for opinion in self.root.findall('.//Opinion'):
            polarity = opinion.get('polarity')

            assert polarity in polarity_counts, f"Polarity {polarity} is not valid"

            polarity_counts[polarity] += 1

        total = sum(polarity_counts.values())

        return {
            'counts': polarity_counts,
            'total': total
        }


if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("PATH_TO_PREPROCESSED_SEMEVAL_14_RESTAURANTS_TRAIN_DATA")

    data_set = DataSet(file_path)
    print(data_set.polarity_frequencies())