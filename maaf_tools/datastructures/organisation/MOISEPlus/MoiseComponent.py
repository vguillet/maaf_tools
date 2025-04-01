
from abc import ABC, abstractmethod
import json


class MoiseComponent(ABC):
    # ============================================================== Properties

    # ============================================================== Check

    # ============================================================== Get

    # ============================================================== Set

    # ============================================================== Add

    # ============================================================== Remove

    # ============================================================== Serialization / Parsing
    # --------------------------------- Abstract Methods ---------------------------------
    @abstractmethod
    def asdict(self) -> dict:
        """
        Create a dictionary containing the fields of the MoiseComponent data class instance with their current values.
        """
        pass

    # --------------------------------- Concrete Methods ---------------------------------
    def to_json(self, indent=2):
        """
        Serializes the MOISEPlus model to a JSON string.

        :return: A JSON-formatted string representation of the model.
        """
        return json.dumps(self.asdict(), indent=indent)

    @classmethod
    def from_json(cls, json_str):
        """
        Creates a MoiseModel instance from a JSON string.

        :return: A new instance of MoiseModel.
        """
        data = json.loads(json_str)
        return cls(data)

    def save_to_file(self, filename: str):
        """
        Saves the MOISEPlus model to a file in JSON format.

        :param filename: The name of the file.
        """
        with open(filename, "w") as f:
            f.write(self.to_json(indent=2))

    @classmethod
    def load_from_file(cls, filename):
        """
        Loads a MOISEPlus model from a JSON file.

        :param filename: The name of the file to load.

        :return: MoiseModel: A new instance of MoiseModel.
        """
        with open(filename, "r") as f:
            data = json.load(f)
        return cls(data)
