import functools

from piecewise.dtype import Classifier
from piecewise.error.classifier_set_error import MemberNotFoundError


def verify_membership(method):
    """Decorator to ensure classifiers in args (both positional and keyword) are
    contained in the classifier set before performing an operation on the set
    with them."""
    @functools.wraps(method)
    def _verify_membership(self, *args, **kwargs):
        all_args = list(args)
        for kwarg in kwargs.values():
            all_args.append(kwarg)
        classifier_args = [
            arg for arg in all_args if isinstance(arg, Classifier)
        ]
        for classifier in classifier_args:
            if classifier not in self._members:
                raise MemberNotFoundError()
        return method(self, *args, **kwargs)

    return _verify_membership


class ClassifierSetBase:
    """Common functionality for ClassifierSet and Population classes.

    Implements __contains__ and __iter__, so is considered to be a both a
    Container and an Iterable. Deliberately does not implement the __len__
    magic method as its notion is confusing. Should length of a classifier set
    correspond to the number of microclassifiers or the number of
    macroclassifiers?

    Instead of causing confusion, the two concepts are explicitly separated:
    see num_micros and num_macros properties.

    Uses a list as the internal data structure to hold Classifier objs, because
    Classifier objs are mutable and thus not hashable, so not suitable for use
    in a hash-based data structure.

    num_micros is an abstract property because the two subclasses of this base
    (ClassifierSet and Population), handle calculation of this property
    differently (namely it is actually calculated for the former but cached
    and validated for the latter).
    """
    def __init__(self):
        self._members = []

    @property
    @abc.abstractmethod
    def num_micros(self):
        raise NotImplementedError

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

    def __eq__(self, other):
        for (my_member, other_member) in zip(self._members, other._members):
            if my_member != other_member:
                return False
        return True
