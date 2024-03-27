
##################################################################################################################

from dataclasses import dataclass, fields, field

try:
    from maaf_tools.datastructures.MaafItem import MaafItem

except:
    from maaf_tools.maaf_tools.datastructures.MaafItem import MaafItem

##################################################################################################################

# @dataclass(frozen=True)
@dataclass
class State(MaafItem):
    timestamp: float           # The timestamp of the state

    def __str__(self) -> str:
        return self.__repr__()
