class Line:
    def __init__(self, first_point, second_point):
        self._first_point = first_point
        self._second_point = second_point
        rise = (second_point.y - first_point.y)
        run = (second_point.x - first_point.x)
        if run == 0.0:
            assert second_point.x == first_point.x
            self._is_vertical = True
            self._m = None
            self._c = None
        else:
            self._is_vertical = False
            self._m = rise / run
            self._c = first_point.y - self._m*first_point.x

    @property
    def is_vertical(self):
        return self._is_vertical

    @property
    def subdomain_min(self):
        return self._first_point.x

    @property
    def subdomain_max(self):
        return self._second_point.x

    def eval(self, x):
        assert not self._is_vertical
        return self._m * x + self._c

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self._first_point!r}, "
                f"{self._second_point!r})")

    def __str__(self):
        return (f"{self._first_point}, {self._second_point}: "
               f"y = {self._m}*x + {self._c}")
