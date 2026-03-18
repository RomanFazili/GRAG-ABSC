import os
from dotenv import load_dotenv
from owlready2 import Ontology, get_ontology


class DataSetOntology:

    def __init__(self, file_path: str):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")
        if not file_path.endswith(".owl"):
            raise ValueError(f"The file {file_path} is not an OWL file")

        self.file_path = file_path

        self.ontology: Ontology = get_ontology(f"file://{os.path.abspath(self.file_path)}").load()


if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv("PATH_TO_RESTAURANT_ONTOLOGY")

    data_set = DataSetOntology(file_path)