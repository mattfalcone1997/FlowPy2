
import copy
from abc import ABC


class BaseGroup(ABC):

    def __init__(self, name, comps):
        self._name = name
        self._comps = comps

    def __iter__(self):
        return self._comps.__iter__()

    def __len__(self):
        return len(self._comps)

    def __contains__(self, val):
        return val in self._comps

    def __getitem__(self, key):
        return self._comps[key]

    def copy(self):
        comps = copy.deepcopy(self._comps)
        return self.__class__(self._name,
                              comps)

    @property
    def name(self) -> str:
        return self._name

    @property
    def comps(self) -> list[str]:
        return self._comps


class Vector(BaseGroup):
    pass


class Tensor2Symmetric(BaseGroup):
    pass


class Tensor2():
    pass
