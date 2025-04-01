
##################################################################################################################

from dataclasses import dataclass, fields, field
from typing import List, Optional, Dict, Any

try:
    from maaf_tools.datastructures.MaafItem import MaafItem

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem

##################################################################################################################


@dataclass
class NestedDict(MaafItem):
    data: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = NestedDict()
        return self.data[key]

    def __setitem__(self, key, value):
        if key not in self.data:
            self.data[key] = NestedDict()

        self.data[key] = value

    # ============================================================== Serialization / Parsing
    def asdict(self) -> dict:
        """
        Create a dictionary containing the fields of the item data class instance with their current values.

        :return: A dictionary with field names as keys and current values.
        """

        data_dict = {}

        for key, value in self.data.items():
            if isinstance(value, NestedDict):
                data_dict[key] = value.asdict()
            else:
                data_dict[key] = value

        return data_dict

    @classmethod
    def from_dict(cls, item_dict: dict):
        """
        Convert a dictionary to an item.

        :param item_dict: The dictionary representation of the item

        :return: An item object
        """

        for key, value in item_dict.items():
            if isinstance(value, dict):
                item_dict[key] = cls().from_dict(value)

        return cls(data=item_dict)