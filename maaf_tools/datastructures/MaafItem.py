
##################################################################################################################

from dataclasses import dataclass, fields, field
from abc import ABC, abstractmethod
from copy import deepcopy

##################################################################################################################


DEBUG = True


@dataclass
class MaafItem(ABC):
    __pre_asdict_subscribers: list[callable] = field(default_factory=list, init=False)

    def __reduce__(self):
        """
        Reduce the item to a dictionary.
        """
        return self.__class__, (*self.asdict(),)  # -> Pass dictionary representation to constructor

    def add_pre_asdict_subscriber(self, subscriber: callable):
        """
        Add a subscriber to the asdict method.
        """
        self.__pre_asdict_subscribers.append(subscriber)

    def call_pre_asdict_subscribers(self):
        """
        Call the pre asdict subscribers.
        """
        for subscriber in self.__pre_asdict_subscribers:
            subscriber(self)

    # ============================================================== Get
    def clone(self):
        """
        Clone the item.
        """
        return self.from_dict(item_dict=deepcopy(self.asdict()))

    # ============================================================== Set

    # ============================================================== Merge
    # @abstractmethod
    # def merge(self, item, prioritise_local: bool = False):
    #     """
    #     Merge the item with another item.
    #     """
    #     pass

    # ============================================================== To
    def asdict(self, include_local: bool = False) -> dict:
        """
        Create a dictionary containing the fields of the Task data class instance with their current values.

        :param include_local: Whether to include the local field in the dictionary.

        :return: A dictionary with field names as keys and current values.
        """
        try:
            from maaf_tools.datastructures.serialisation import asdict

        except ImportError:
            from maaf_tools.maaf_tools.datastructures.serialisation import asdict

        # -> Call pre asdict subscribers
        self.call_pre_asdict_subscribers()

        if not include_local:
            fields_exclusion_lst = ["local"]
        else:
            fields_exclusion_lst = []

        fields_dict = asdict(
            item=self,
            fields_exclusion_lst=fields_exclusion_lst
        )

        return fields_dict

    # ============================================================== From
    @classmethod
    def from_dict(cls, item_dict: dict, partial: bool = False) -> object:
        """
        Convert a dictionary to a task.

        :param item_dict: The dictionary representation of the task
        :param partial: Whether to allow creation from a dictionary with missing fields.

        :return: A task object
        """
        try:
            from maaf_tools.datastructures.serialisation import from_dict

        except ImportError:
            from maaf_tools.maaf_tools.datastructures.serialisation import from_dict

        item = from_dict(
            cls=cls,
            item_dict=item_dict,
            partial=partial,
            fields_exclusion_lst=["local"]
        )

        return item
