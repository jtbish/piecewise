import math
import pickle


def main():
    with open("pop.pkl", "rb") as fp:
        pop = pickle.load(fp)
    pop = [classifier for classifier in pop]
    fitness_sorted_pop = sorted(pop,
                                key=lambda classifier: classifier.fitness,
                                reverse=True)
    for i in range(20):
        pretty_print_classifier(fitness_sorted_pop[i])


def pretty_print_classifier(classifier):
    _print("[ ")
    idxs = range(len(classifier.condition))
    for i in idxs:
        pretty_print_elem(classifier.condition[i])
        if i != idxs[-1]:
            _print(", ")
    _print(" ] -> ")
    _print(f"{classifier.action}: {classifier.prediction:.2f}, "
           f"{classifier.error:.2f}, "
           f"{classifier.fitness:.2f}, "
           f"{classifier.numerosity}")
    _print("\n")


def pretty_print_elem(elem):
    bucket_width = 0.1
    interval_min = 0.0
    interval_max = 1.0
    total_buckets = 10

    # truncate elem lower and upper to fit in interval
    elem_lower = float(max(elem.lower(), interval_min))
    elem_upper = float(min(elem.upper(), interval_max))

    lhs_unfilled_buckets = (elem_lower - interval_min) / bucket_width
    lhs_has_partial_bucket = lhs_unfilled_buckets % 1 != 0
    rhs_unfilled_buckets = (interval_max - elem_upper) / bucket_width
    rhs_has_partial_bucket = rhs_unfilled_buckets % 1 != 0

    num_full_center_buckets = total_buckets - \
        (math.ceil(lhs_unfilled_buckets) + math.ceil(rhs_unfilled_buckets))

    count = 0
    for _ in range(math.floor(lhs_unfilled_buckets)):
        _print("-")
        count += 1
    if lhs_has_partial_bucket:
        _print("o")
        count += 1
    for _ in range(num_full_center_buckets):
        _print("O")
        count += 1
    if rhs_has_partial_bucket:
        _print("o")
        count += 1
    for _ in range(math.floor(rhs_unfilled_buckets)):
        _print("-")
        count += 1
    #assert count == total_buckets


def _print(str_):
    # print with no newline
    print(str_, end='')


if __name__ == "__main__":
    main()
