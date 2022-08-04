import numpy as np
Array = np.ndarray


def to_hashable(x):
    r"""Converts to hashable data type."""
    if isinstance(x, list):
        return HashableList(x)
    if isinstance(x, tuple):
        return HashableTuple(x)
    if isinstance(x, set):
        return HashableSet(x)
    if isinstance(x, dict):
        return HashableDict(x)
    if isinstance(x, Array):
        return HashableArray(x)

    try:
        hash(x)
    except:
        raise TypeError("Hashable type is not implemented.")
    else:
        return x


def _is_custom_hashable(val):
    r"""Returns whether the input is a custom hashable type."""
    return (
        isinstance(val, HashableList)
        or isinstance(val, HashableTuple)
        or isinstance(val, HashableSet)
        or isinstance(val, HashableDict)
        or isinstance(val, HashableArray)
    )


class HashableList(list):
    r"""Hashable list class."""

    def __init__(self, native_list: list):
        super(HashableList, self).__init__([to_hashable(val) for val in native_list])

    def __hash__(self):
        return hash(tuple(self))

    def native(self) -> list:
        converted = []
        for val in self:
            converted.append(val.native() if _is_custom_hashable(val) else val)
        return converted


class HashableTuple(tuple):
    r"""Hashable tuple class."""

    def __new__(cls, native_tuple: tuple):
        return super(HashableTuple, cls).__new__(cls, tuple(to_hashable(val) for val in native_tuple))

    def native(self) -> tuple:
        converted = []
        for val in self:
            converted.append(val.native() if _is_custom_hashable(val) else val)
        return tuple(converted)


class HashableSet(set):
    r"""Hashable set class."""

    def __init__(self, native_set: set):
        super(HashableSet, self).__init__([to_hashable(val) for val in native_set])

    def __hash__(self):
        return hash(frozenset(self))

    def native(self) -> set:
        converted = set()
        for val in self:
            converted.add(val.native() if _is_custom_hashable(val) else val)
        return converted


class HashableDict(dict):
    r"""Hashable dictionary class."""

    def __init__(self, native_dict: dict):
        super(HashableDict, self).__init__({key: to_hashable(val) for key, val in native_dict.items()})

    def __hash__(self):
        return hash(frozenset(self.items()))

    def native(self) -> dict:
        converted = {}
        for key, val in self.items():
            converted[key] = val.native() if _is_custom_hashable(val) else val
        return converted


class HashableArray(HashableList):
    r"""Hashable numpy array implemented by HashableList."""

    def __init__(self, native_array: Array):
        self.shape = native_array.shape
        self.dtype = native_array.dtype
        vals = list(native_array.reshape(-1))
        super(HashableArray, self).__init__(vals)

    def native(self) -> Array:
        return np.array(self, dtype=self.dtype).reshape(self.shape)
