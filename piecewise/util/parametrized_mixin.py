import abc


class ParametrizedMixin(metaclass=abc.ABCMeta):
    _parametrization_dict = None

    def record_parametrization(self, **kwargs):
        self._parametrization_dict = kwargs

    def get_parametrization_as_str(self):
        return f"{self.__class__.__name__}: {self._parametrization_dict}"
