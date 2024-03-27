
##################################################################################################################

from dataclasses import dataclass, fields, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
from abc import ABC, abstractmethod

##################################################################################################################


DEBUG = True


class MaafItem(ABC):
    def __reduce__(self):
        """
        Reduce the item to a dictionary.
        """
        return self.__class__, (*self.asdict(),)  # -> Pass dictionary representation to constructor

    @abstractmethod
    def asdict(self) -> dict:
        """
        Create a dictionary containing the fields of the item data class instance with their current values.

        :return: A dictionary with field names as keys and current values.
        """
        pass

    # @abstractmethod
    # def asdf(self) -> pd.DataFrame:
    #     """
    #     Create a pandas DataFrame from the item data class instance.
    #     """
    #     pass

    @classmethod
    @abstractmethod
    def from_dict(cls, item_dict: dict, partial: bool):
        """
        Convert a dictionary to an item.

        :param item_dict: The dictionary representation of the item
        :param partial: Whether to allow creation from a dictionary with missing fields.

        :return: An item object
        """
        pass
