class Rule:
    """Represents a rule (condition to action mapping) for a classifier."""
    def __init__(self, condition, action):
        self._condition = condition
        self._action = action

    @property
    def condition(self):
        return self._condition

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, new_action):
        self._action = new_action

    def __eq__(self, other):
        return self._condition == other.condition and \
               self._action == other.action

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self._condition!r}, {self._action!r})")

    def __str__(self):
        return f"{self._condition} -> {self._action}"
