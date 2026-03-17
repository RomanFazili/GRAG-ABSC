import os

class OntologyImplementation:

    def __init__(self, file_path: str):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")
        if not file_path.endswith(".owl"):
            raise ValueError(f"The file {file_path} is not an OWL file")

        self.file_path = file_path



if __name__ == "__main__":
    file_path = input("Enter the path to the XML file: ")

    ontology_implementation = OntologyImplementation(file_path)
    print(ontology_implementation)