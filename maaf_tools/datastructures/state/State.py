
##################################################################################################################

from dataclasses import dataclass, fields, field

try:
    from maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.Singleton import SLogger

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem
    from maaf_tools.maaf_tools.Singleton import SLogger

##################################################################################################################


# @dataclass(frozen=True)
@dataclass
class State(MaafItem):
    _timestamp: float
    __get_timestamp: callable = field(default=None, init=False)

    def __str__(self) -> str:
        return self.__repr__()

    def print_get_timestamp(self) -> None:
        return self.__get_timestamp
        # SLogger().info(f"Get timestamp callable: {self.__get_timestamp}")

    def set_get_timestamp(self, get_timestamp: callable) -> None:
        self.__get_timestamp = get_timestamp

    @property
    def timestamp(self) -> float:
        if self.__get_timestamp is not None:
            self._timestamp = self.__get_timestamp()

        return self._timestamp
