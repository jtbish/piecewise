import abc
from collections import namedtuple

from .line import Line

RANGE_MIN = 0.0
RANGE_MAX = 1.0

Point = namedtuple("Point", ["x", "y"])


class MembershipFuncABC(metaclass=abc.ABCMeta):
    """Fuzzy set / membership function."""
    def __init__(self, domain, name):
        self._domain = domain
        self._name = name

    @property
    def domain(self):
        return self._domain

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def fuzzify(self, input_scalar):
        raise NotImplementedError


class PiecewiseLinearMembershipFunc(MembershipFuncABC):
    """Fuzzy set / membership function composed of a series of linear sections,
    e.g. triangle or trapezoid."""
    def __init__(self, domain, points, name):
        super().__init__(domain, name)
        self._points = points
        self._lines = self._create_lines(self._points)

    def _create_lines(self, points):
        lines = self._create_lines_from_points(points)
        lines = self._remove_vertical_lines(lines)
        return lines

    def _create_lines_from_points(self, points):
        lines = []
        self._add_flat_line_at_domain_start_if_needed(points, lines)
        self._add_lines_for_given_points(points, lines)
        self._add_flat_line_at_domain_end_if_needed(points, lines)
        return lines

    def _add_flat_line_at_domain_start_if_needed(self, points, lines):
        first_point = points[0]
        assert first_point.y == RANGE_MIN
        flat_line_needed = first_point.x > self._domain.min
        if flat_line_needed:
            lines.append(Line(Point(self._domain.min, RANGE_MIN), first_point))

    def _add_lines_for_given_points(self, points, lines):
        line_lhs_points = points[:-1]
        line_rhs_points = points[1:]
        for (lhs_point, rhs_point) in zip(line_lhs_points, line_rhs_points):
            lines.append(Line(lhs_point, rhs_point))

    def _add_flat_line_at_domain_end_if_needed(self, points, lines):
        last_point = points[-1]
        assert last_point.y == RANGE_MIN
        flat_line_needed = last_point.x < self._domain.max
        if flat_line_needed:
            lines.append(Line(last_point, Point(self._domain.max, RANGE_MIN)))

    def _remove_vertical_lines(self, lines):
        return [line for line in lines if line.is_vertical]

    def fuzzify(self, input_scalar):
        assert self._domain.min <= input_scalar <= self._domain.max
        result = None
        for line in self._lines:
            if line.subdomain_min <= input_scalar <= line.subdomain_max:
                result = line.eval(input_scalar)
                break
        assert result is not None and RANGE_MIN <= result <= RANGE_MAX
        return result


def make_triangular_membership_func(domain, base_lhs_x, apex_x, base_rhs_x,
                                    name):
    assert domain.min <= base_lhs_x <= apex_x <= base_rhs_x <= domain.max

    points = [
        Point(base_lhs_x, RANGE_MIN),
        Point(apex_x, RANGE_MAX),
        Point(base_rhs_x, RANGE_MIN),
    ]
    return PiecewiseLinearMembershipFunc(domain, points, name)


def make_trapezoidal_membership_func(domain, bottom_base_lhs_x, top_base_lhs_x,
                                     top_base_rhs_x, bottom_base_rhs_x, name):
    assert domain.min <= bottom_base_lhs_x <= top_base_lhs_x <= \
            top_base_rhs_x <= bottom_base_rhs_x <= domain.max

    points = [
        Point(bottom_base_lhs_x, RANGE_MIN),
        Point(top_base_lhs_x, RANGE_MAX),
        Point(top_base_rhs_x, RANGE_MAX),
        Point(bottom_base_rhs_x, RANGE_MIN),
    ]
    return PiecewiseLinearMembershipFunc(domain, points, name)
