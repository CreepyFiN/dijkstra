"""Checks for the bucket tuner ranking policy; no GPU or graph data required."""
import math

from python.tune_delta import scores


def test_case_normalization_and_median_reject_one_fast_outlier():
    rows=[]
    for case, scale in (("near",1),("far",1000)):
        for width, times in ((100,[1,4,4]),(1000,[2,2,2])):
            for value in times:
                rows.append(dict(case=case,delta=width,backend="delta-cpu",mode="early",
                                 algorithm_ms=value*scale))
    # The long route has equal weight, not 1000 times the influence. A lucky
    # one-off result cannot make the consistently slower width win.
    ranked=scores(rows,"delta-cpu","early")
    assert [row["delta"] for row in ranked]==[1000,100]
    assert math.isclose(ranked[0]["normalized_score"],1)
    assert math.isclose(ranked[1]["normalized_score"],2)


def test_incomplete_candidates_and_unrelated_backends_are_excluded():
    rows=[dict(case=case,delta=100,backend="delta-cuda",mode="full",algorithm_ms=10)
          for case in ("a","b")]
    rows += [dict(case="a",delta=1,backend="delta-cuda",mode="full",algorithm_ms=0.01),
             dict(case="other",delta=1,backend="delta-cpu",mode="early",algorithm_ms=0.01)]
    assert [row["delta"] for row in scores(rows,"delta-cuda","full")]==[100]
    assert scores(rows,"missing","full")==[]
