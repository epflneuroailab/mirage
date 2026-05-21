"""Training metrics.

Predictive encoding metrics are re-exported here for the historical
``brain_enc.training.metrics`` import path. Add representation metrics such as
RSA and CKA in sibling modules.
"""

from brain_enc.training.metrics.predictive import (
    ExplainedVariance,
    GroupedExplainedVariance,
    GroupedPearsonR,
    GroupedWeightedMean,
    PearsonR,
    WeightedMean,
)
from brain_enc.training.metrics.cka import (
    CenteredKernelAlignment,
    CenteredKernelAlignmentTorch,
)
from brain_enc.training.metrics.rsa import (
    RepresentationalSimilarityAnalysis,
    RepresentationalSimilarityAnalysisTorch,
)

__all__ = [
    "CenteredKernelAlignment",
    "CenteredKernelAlignmentTorch",
    "ExplainedVariance",
    "GroupedExplainedVariance",
    "GroupedPearsonR",
    "GroupedWeightedMean",
    "PearsonR",
    "RepresentationalSimilarityAnalysis",
    "RepresentationalSimilarityAnalysisTorch",
    "WeightedMean",
]
