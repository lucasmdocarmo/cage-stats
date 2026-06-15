"""
Prometheus exposition-text parser.

Converts the raw text returned by a vLLM ``/metrics`` endpoint into a
``Families`` dict that maps each sample name to its list of ``(labels, value)``
pairs.  All aggregation (summing across label sets, extracting histogram
buckets) happens here so the metrics engine works with clean numeric data.

``Families`` type alias
    ``dict[str, list[tuple[dict[str, str], float]]]``
    Key  = Prometheus sample name (e.g. ``"vllm:generation_tokens_total"``).
    Value = ordered list of ``(label_dict, float_value)`` tuples.

Helper functions
----------------
``sum_value(families, name)``
    Sum all values for a sample name across label sets.  Returns ``None`` when
    the metric is absent.

``first_value(families, name)``
    Return the first value for a sample name (useful for gauge metrics that
    carry a single value).

``info_labels(families, name)``
    Return the label dict of the first sample — used for vLLM info metrics
    (e.g. ``vllm:cache_config_info``).

``get_buckets(families, base)``
    Aggregate ``<base>_bucket`` samples into a sorted ``[(le, cum_count)]``
    list, summing counts across label sets with the same ``le``.

``hist_count / hist_sum``
    Convenience wrappers for the ``_count`` / ``_sum`` siblings of a histogram.
"""

from __future__ import annotations

from prometheus_client.parser import text_string_to_metric_families

Families = dict[str, list[tuple[dict[str, str], float]]]


def parse_metrics(text: str) -> Families:
    families: Families = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            families.setdefault(sample.name, []).append((dict(sample.labels), sample.value))
    return families


def sum_value(families: Families, name: str) -> float | None:
    rows = families.get(name)
    if not rows:
        return None
    return sum(v for _, v in rows)


def first_value(families: Families, name: str) -> float | None:
    rows = families.get(name)
    if not rows:
        return None
    return rows[0][1]


def info_labels(families: Families, name: str) -> dict[str, str]:
    rows = families.get(name)
    return rows[0][0] if rows else {}


def get_buckets(families: Families, base: str) -> list[tuple[float, float]]:
    rows = families.get(base + "_bucket", [])
    agg: dict[float, float] = {}
    for labels, value in rows:
        le = float(labels["le"])
        agg[le] = agg.get(le, 0.0) + value
    return sorted(agg.items())


def hist_count(families: Families, base: str) -> float | None:
    return sum_value(families, base + "_count")


def hist_sum(families: Families, base: str) -> float | None:
    return sum_value(families, base + "_sum")
