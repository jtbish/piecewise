import abc
import functools

from piecewise.error.classifier_set_error import MemberNotFoundError


def verify_membership(method):
    """Decorator to ensure classifiers in positional args are contained in the
    classifier set before performing an operation on the set with them."""
    @functools.wraps(method)
    def _verify_membership(self, *args, **kwargs):
        for classifier in args:
            if classifier not in self._members:
                raise MemberNotFoundError()
        return method(self, *args, **kwargs)

    return _verify_membership


class AbstractClassifierSet(metaclass=abc.ABCMeta):
    """An abstract container for storage of Classifier objects.

    Implements __contains__ and __iter__, so is considered to be a both a
    Container and an Iterable. Deliberately does not implement the __len__
    magic method as its notion is confusing. Should length of a classifier set
    correspond to the number of microclassifiers or the number of
    macroclassifiers?

    Instead of causing confusion, the two concepts are explicitly separated
    into different methods: see num_micros() and num_macros().

    Uses a list as the internal data structure to hold Classifier objs, because
    Classifier objs are mutable and thus not hashable, so not suitable for use
    in a hash based data structure.
    """
    def __init__(self):
        self._members = []
        self._num_micros = 0

    @property
    def num_micros(self):
        """Originally was a listcomp that returned sum of member numerosities.
        Was quite slow (unsurprisingly) when profiled, so was changed to simple
        attribute lookup.

        Now subclasses are responsible for incrementing/decrementing the
        attribute via two methods below.

        This change caused 30% latency reduction in execution time."""
        return self._num_micros

    def _inc_num_micros(self, added_numerosity):
        self._num_micros += added_numerosity

    def _dec_num_micros(self, removed_numerosity):
        self._num_micros -= removed_numerosity
        assert self._num_micros >= 0

    @property
    def num_macros(self):
        return len(self._members)

    def __contains__(self, member):
        return member in self._members

    def __iter__(self):
        return iter(self._members)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._members!r})"

    def __str__(self):
        return "{ " + ",\n".join([str(member) for member in self._members]) \
                + " }"
