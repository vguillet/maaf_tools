
##################################################################################################################

from dataclasses import dataclass, fields, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod
import pandas as pd

try:
    from maaf_tools.datastructures.MaafItem import MaafItem

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem

##################################################################################################################


DEBUG = False


@dataclass
class MaafList(MaafItem):
    items: list = field(default_factory=list)
    item_class = None

    __on_add_item_listeners: list[callable] = field(default_factory=list)
    __on_update_item_listeners: list[callable] = field(default_factory=list)
    __on_remove_item_listeners: list[callable] = field(default_factory=list)
    __on_edit_list_listeners: list[callable] = field(default_factory=list)
    manual_on_edit_list_listeners_call = False

    def __str__(self):
        return self.__repr__()

    def __contains__(self, item: str or item_class) -> bool:
        """
        Check if an item exists in the item log.

        :param item: The item to check for. Can be either an item object or an item id.
        """

        if isinstance(item, str):
            return item in self.ids
        else:
            return item in self.items

    def __len__(self) -> int:
        """
        Retrieve number of items in list
        """
        return len(self.items)

    def __iter__(self) -> iter:
        """
        Retrieve an iterator for the item log.
        """
        # -> Sort items list by id
        self.sort(key=lambda x: x.id)

        return iter(self.items)

    def __getitem__(self, item_id: int or str) -> Optional[item_class]:
        """
        Retrieve an item from the item list by its id as index.
        """
        # -> Find the item with the given ID
        for item in self.items:
            # > If the item exists, return it
            if item.id == item_id:
                return item

        # > If the item does not exist, warn the user and return None
        if DEBUG: print(
            f"!!! Get item by index failed: {self.item_class} with id '{item_id}' does not exist in the item log !!!")
        return None

    def __reduce__(self):
        """
        Reduce the item log
        """
        return self.__class__, (self.items,)

    # ============================================================== Properties
    # ------------------------------ IDs
    @property
    def ids(self) -> list[int]:
        """
        Get a list of item ids in the item log.
        """
        return [item.id for item in self.items]

    # ============================================================== Listeners
    # ------------------------------ Edit
    def add_on_edit_list_listener(self, listener: callable) -> None:
        """
        Add a listener to be called when the item list is edited.

        :param listener: The listener to add.
        """
        self.__on_edit_list_listeners.append(listener)

    def call_on_edit_list_listeners(self, item: item_class) -> None:
        """
        Call all on_edit_list listeners.
        """

        # -> If the manual_on_edit_list_listeners_call flag is set, return
        if self.manual_on_edit_list_listeners_call:
            return

        # -> Call all on_edit_list listeners
        for listener in self.__on_edit_list_listeners:
            listener(item)

    # ------------------------------ Update
    def add_on_update_item_listener(self, listener: callable) -> None:
        """
        Add a listener to be called when an item is updated in the item log.

        :param listener: The listener to add.
        """
        self.__on_update_item_listeners.append(listener)

    def call_on_update_item_listeners(self, item: item_class) -> None:
        """
        Call all on_update_item listeners.

        :param item: The item that was updated.
        """
        # -> Call all on_update_item listeners
        for listener in self.__on_update_item_listeners:
            listener(item)

        # -> Call all on_edit_list listeners
        self.call_on_edit_list_listeners(item)

    # ------------------------------ Add
    def add_on_add_item_listener(self, listener: callable) -> None:
        """
        Add a listener to be called when an item is added to the item log.

        :param listener: The listener to add.
        """
        self.__on_add_item_listeners.append(listener)

    def call_on_add_item_listeners(self, item: item_class) -> None:
        """
        Call all on_add_item listeners.

        :param item: The item that was added.
        """
        # -> Call all on_add_item listeners
        for listener in self.__on_add_item_listeners:
            listener(item)

        # -> Call all on_edit_list listeners
        self.call_on_edit_list_listeners(item)

    # ------------------------------ Remove
    def add_on_remove_item_listener(self, listener: callable) -> None:
        """
        Add a listener to be called when an item is removed from the item log.

        :param listener: The listener to add.
        """
        self.__on_remove_item_listeners.append(listener)

    def call_on_remove_item_listeners(self, item: item_class) -> None:
        """
        Call all on_remove_item listeners.

        :param item: The item that was removed.
        """
        # -> Call all on_remove_item listeners
        for listener in self.__on_remove_item_listeners:
            listener(item)

        # -> Call all on_edit_list listeners
        self.call_on_edit_list_listeners(item)

    # ============================================================== Sort
    def sort(self, key: callable = None, reverse: bool = False) -> None:
        """
        Sort the items in the item log.

        :param key: The key to sort the items by.
        :param reverse: Whether to sort the items in reverse order.
        """
        self.items.sort(key=key, reverse=reverse)

    # ============================================================== Get

    # ============================================================== Set
    def update_item_fields(self,
                           item: int or str or item_class,
                           field_value_pair: dict,
                           add_to_local: bool = False,
                           add_to_shared: bool = False
                           ) -> None:
        """
        Update fields of an item with given item_id.
        Returns True if item is updated successfully, False otherwise.

        :param item: The item to update.
        :param field_value_pair: A dict containing the field and value to update.
        :param add_to_local: Whether to add the field to the local field if it does not exist.
        :param add_to_shared: Whether to add the field to the shared field if it does not exist.
        """

        assert add_to_local + add_to_shared < 2, "Cannot add to both local and shared fields"

        # -> If item is a string or int, find the item with the given ID
        if isinstance(item, str) or isinstance(item, int):
            item = self[item]

        # -> If item is an item object, find the item with the given ID
        else:
            item = self[item.id]

        if item is None:
            raise ValueError(f"!!! Update item fields failed: {self.item_class} with id '{item}' does not exist in the item log !!!")
            return None

        # If the item does not exist, warn the user
        elif item.id not in self.ids:
            raise ValueError(f"!!! Update item fields failed: {self.item_class} with id '{item.id}' does not exist in the item log !!!")
            return

        # -> Update the fields of the item
        for key, value in field_value_pair.items():
            # > Update the field if it exists
            if hasattr(item, key):
                setattr(item, key, value)

            # > Check if the field exist in the shared field
            elif hasattr(item.shared, key):
                setattr(item.shared, key, value)

            # > Check if the field exist in the local field
            elif hasattr(item.local, key):
                setattr(item.local, key, value)

            # > Add the field to the local field if it does not exist and add_to_local is True
            elif add_to_local:
                item.local[key] = value

            # > Add the field to the shared field if it does not exist and add_to_shared is True
            elif add_to_shared:
                item.shared[key] = value

            else:
                raise ValueError(f"!!! Update item fields failed (3): {self.item_class} with id '{item.id}' does not have field '{key}' !!!")

        # -> Call the on_update_item method
        self.call_on_update_item_listeners(item)

    # # ============================================================== Merge
    # def merge(self, item_list: "MaafList", prioritise_local: bool = False) -> None:
    #     """
    #     Merge the items of another item list into the item list.
    #
    #     :param item_list: The item log to merge into the item log.
    #     :param prioritise_local: Whether to prioritise the local field of the item over the shared field.
    #     """
    #
    #     # -> Check if the item log is of the same type
    #     if self.item_class != item_list.item_class:
    #         raise ValueError(f"!!! Merge item log failed: Merging item list with different item class '{item_list.item_class}' into item list with item class '{self.item_class}' !!!")
    #
    #     # -> For all items not in item list
    #     for item_id in item_list.ids:
    #         if item_id not in self.ids:
    #             self.add_item(item_list[item_id])
    #
    #         else:
    #             self[item_id].merge(item_list[item_id], prioritise_local)
    #
    #         elif not prioritise_local:
    #             local_item = self[item_id]
    #             new_item = item_list[item_id]               # > retrieve the new item
    #             new_item.local = local_item.local           # > add the local field of the old item to the new item
    #
    #             self.remove_item(local_item)                # > remove the old item
    #             self.add_item(new_item)                     # > add the new item

    # ============================================================== Add
    def add_item(self, item: item_class) -> bool:
        """
        Add an item to the item log. If the item is a list, add each item to the item log individually recursively.

        :param item: The item to add to the item log.
        """
        # -> Check if the item already exists in the item log
        # > If the item already exists, skip and warn user
        if item in self.items:
            if DEBUG: print(f"!!! Add item failed: {self.item_class} with id '{item.id}' already exists in the item log !!!")

            return False

        # > else, add the item to the item log
        else:
            # Insert item in the correct position (order by increasing id)
            for i, item_ in enumerate(self.items):
                if item_.id > item.id:
                    self.items.insert(i, item)
                    break

            else:
                self.items.append(item)

            # self.items.append(item)

        # -> Sort items list by id
        # self.sort(key=lambda x: x.id)

        # -> Call the on_add_item method
        self.call_on_add_item_listeners(item)

        return True

    def add_item_by_dict(self, item_data: dict) -> bool:
        """
        Add a item to the item log using a dictionary.

        :param item_data: A dictionary containing the fields of the item.
        """

        # -> Convert the dictionary to an item object
        new_item = self.item_class.from_dict(item_data)

        # -> Add the item to the item log
        return self.add_item(new_item)

    # ============================================================== Remove
    def remove_item(self, item: item_class) -> bool:
        """
        Remove an item from the item log.

        :param item: The item to remove from the item log.
        """

        # -> Remove the item from the item log
        try:
            self.items.remove(item)
        except ValueError:
            if DEBUG: print(f"!!! Remove item failed: {self.item_class} with id '{item.id}' does not exist in the item log !!!")

            return False

        # -> Call the on_remove_item method
        self.call_on_remove_item_listeners(item)

        return True

    def remove_item_by_id(self, item_id: str or int) -> None:
        """
        Remove a item from the item log.

        :param item_id: The id of the item to remove from the item log.
        """
        # -> Find the item with the given ID
        item = next((t for t in self.items if t.id == item_id), None)

        # -> If the item exists, remove it from the item log
        if item:
            self.remove_item(item)
        else:
            if DEBUG: print(f"!!! Remove item by id failed: {self.item_class} with id '{item_id}' does not exist in the item log !!!")

    # ============================================================== To
    # def asdict(self) -> dict:
    #     """
    #     Create a dictionary containing the fields of the maaflist data class instance with their current values.
    #
    #     :return: A dictionary with field names as keys and current values.
    #     """
    #     # -> Get the fields of the maaflist class
    #     maaflist_fields = fields(self)
    #
    #     # -> Exclude all fields with __ in the name
    #     maaflist_fields = [f for f in maaflist_fields if "__" not in f.name]
    #
    #     # -> Exclude the local field
    #     maaflist_fields = [f for f in maaflist_fields if f.name != "local"]
    #
    #     # -> Exclude the items field
    #     maaflist_fields = [f for f in maaflist_fields if f.name != "items"]
    #
    #     # -> Create a dictionary with field names as keys and the value as values
    #     fields_dict = {}
    #
    #     for field in maaflist_fields:
    #         # > If field value has asdict method, call it
    #         if hasattr(getattr(self, field.name), "asdict"):
    #             fields_dict[field.name] = getattr(self, field.name).asdict()
    #
    #         # > Else, add the field value to the dictionary
    #         else:
    #             fields_dict[field.name] = getattr(self, field.name)
    #
    #     # -> Add the items to the dictionary
    #     fields_dict["items"] = [item.asdict() for item in self.items]
    #
    #     return fields_dict

    def asdf(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the items in the item list.

        :return: A pandas DataFrame with the items in the item list.
        """
        # -> Get the fields of the item class
        item_fields = fields(self.item_class)

        # -> Exclude all fields with __ in the name
        item_fields = [f for f in item_fields if "__" not in f.name]

        # -> Create a dictionary with field names as keys and the value as values
        items_df = pd.DataFrame(columns=[f.name for f in item_fields])

        for item in self.items:
            # > Create a dictionary with field names as keys and the value as values
            item_dict = {}

            for field in item_fields:
                # > If field value has asdict method, call it
                if hasattr(getattr(item, field.name), "asdict"):
                    field_dict = getattr(item, field.name).asdict()

                    for key, value in field_dict.items():
                        if "__" not in key:
                            item_dict[key] = value

                # > Else, add the field value to the dictionary
                else:
                    item_dict[field.name] = getattr(item, field.name)

            # > Add the dictionary as new row to the DataFrame
            items_df = items_df._append(item_dict, ignore_index=True)

        # -> Create a pandas DataFrame from the dictionary
        return items_df

    # ============================================================== From
    # @classmethod
    # def from_dict(cls, maaflist_dict: dict, partial=False) -> "MaafList":
    #     """
    #     Construct a maaflist from a dictionary.
    #
    #     :param maaflist_dict: The dictionary representation of the maaflist.
    #     :param partial: Whether to allow creation from a dictionary with missing fields.
    #     """
    #
    #     # -> Get the fields of the maaflist class
    #     maaflist_fields = fields(cls)
    #
    #     # -> Exclude all fields with __ in the name
    #     maaflist_fields = [f for f in maaflist_fields if "__" not in f.name]
    #
    #     # -> Exclude the local field
    #     maaflist_fields = [f for f in maaflist_fields if f.name != "local"]
    #
    #     # -> Exclude the items field
    #     maaflist_fields = [f for f in maaflist_fields if f.name != "items"]
    #
    #     # -> Extract field names from the fields
    #     field_names = {f.name for f in maaflist_fields}
    #
    #     if not partial:
    #         # -> Check if all required fields are present in the dictionary
    #         if not field_names.issubset(maaflist_dict.keys()):
    #             raise ValueError(f"!!! MAAFList creation from dictionary failed: MAAFList dictionary is missing required fields: {maaflist_dict.keys() - field_names} !!!")
    #
    #     else:
    #         # > Remove fields not present in the dictionary
    #         maaflist_fields = [f for f in maaflist_fields if f.name in maaflist_dict]
    #
    #     # -> Extract values from the dictionary for the fields present in the class
    #     field_values = {}
    #
    #     for field in maaflist_fields:
    #         # > If field value has from_dict method, call it
    #         if hasattr(field.type, "from_dict"):
    #             field_values[field.name] = field.type.from_dict(maaflist_dict[field.name])
    #         else:
    #             field_values[field.name] = maaflist_dict[field.name]
    #
    #     # -> Create class instance
    #     maaflist = cls(**field_values)
    #
    #     # -> Create items from the dictionary items
    #     for item_dict in maaflist_dict["items"]:
    #         maaflist.add_item_by_dict(item_dict)
    #
    #     # -> Create and return a Task object
    #     return maaflist
