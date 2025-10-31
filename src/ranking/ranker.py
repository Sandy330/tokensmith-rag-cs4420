"""
ranker.py

This module supports ranking strategies applied after chunk retrieval.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Any

# typedef Candidate as base, we might change this into a class later
# Each candidate is identified by its global index into `chunks`
Candidate = int


def _normalize_list(xs: List[float]) -> List[float]:
    if not xs:
        return xs
    s = float(sum(xs))
    if s <= 0.0:
        # avoid div-by-zero; fall back to equal weights later
        return xs
    return [float(x) / s for x in xs]


def _normalize_dict(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return d
    s = float(sum(d.values()))
    if s <= 0.0:
        return d
    return {k: float(v) / s for k, v in d.items()}


class EnsembleRanker:
    """
    Computes weighted reciprocal rank fusion (RRF) or weighted linear fusion of
    normalized retriever scores.

    ensemble_method: 'linear' or 'rrf'
    weights:
      - dict: {"faiss": 0.6, "bm25": 0.4}
      - list: [0.6, 0.4]  (applied by retriever order in raw_scores)
      - None: equal weights
    """

    def __init__(self, ensemble_method: str, weights: Any = None, rrf_k: int = 60):
        self.ensemble_method = (ensemble_method or "rrf").lower().strip()
        self.rrf_k = int(rrf_k)

        # accept dict/list/None; normalize (sum -> 1.0) when possible
        self._weights_dict: Dict[str, float] = {}
        self._weights_list: List[float] = []

        if isinstance(weights, dict):
            self._weights_dict = _normalize_dict({str(k): float(v) for k, v in weights.items()})
        elif isinstance(weights, (list, tuple)):
            self._weights_list = _normalize_list([float(v) for v in weights])
        elif weights is None:
            pass  # equal weights later
        else:
            # unknown type → ignore; equal weights later
            pass

    def _resolve_weights(self, raw_scores: Dict[str, Dict[Candidate, float]]) -> Dict[str, float]:
        """
        Build a weights map per retriever name for the current call, using:
          1) dict by name if provided,
          2) list by order if provided,
          3) else equal weights.
        """
        names: List[str] = list(raw_scores.keys())

        # 1) explicit dict by name
        if self._weights_dict:
            # ensure every name has a weight; missing -> equal share fallback
            missing = [n for n in names if n not in self._weights_dict]
            if missing:
                # equal share for missing, then renormalize
                equal = 1.0 / max(len(names), 1)
                merged = {n: self._weights_dict.get(n, equal) for n in names}
                return _normalize_dict(merged)
            return _normalize_dict({n: self._weights_dict.get(n, 0.0) for n in names})

        # 2) list by order
        if self._weights_list:
            if len(self._weights_list) < len(names):
                # pad with equal shares for the remainder
                pad = [1.0] * (len(names) - len(self._weights_list))
                ws = _normalize_list(list(self._weights_list) + pad)
            else:
                ws = _normalize_list(list(self._weights_list[:len(names)]))
            return {n: float(ws[i]) for i, n in enumerate(names)}

        # 3) equal weights
        if not names:
            return {}
        eq = 1.0 / len(names)
        return {n: eq for n in names}

    def rank(self, raw_scores: Dict[str, Dict[Candidate, float]]) -> List[int]:
        """
        Executes the rank fusion process on the provided raw scores.

        raw_scores format:
          { retriever_name: { candidate_id: score, ... }, ... }
        """
        if not raw_scores:
            return []

        weights = self._resolve_weights(raw_scores)

        # Collect scores from each active retriever with weight > 0
        per_retriever_scores: Dict[str, Dict[Candidate, float]] = {}
        for name, scores in raw_scores.items():
            w = float(weights.get(name, 0.0))
            if w > 0.0 and scores:
                per_retriever_scores[name] = scores
                # TODO: retrieval/ranker logging can go here

        if not per_retriever_scores:
            return []

        # Fuse scores using the specified method
        if self.ensemble_method == "rrf":
            ordered = self._weighted_rrf_fuse(per_retriever_scores, weights)
        elif self.ensemble_method == "linear":
            ordered = self._weighted_linear_fuse(per_retriever_scores, weights)
        else:
            raise NotImplementedError(f"Ranking method '{self.ensemble_method}' is not implemented.")

        return ordered

    def _weighted_rrf_fuse(
        self,
        per_retriever_scores: Dict[str, Dict[Candidate, float]],
        weights: Dict[str, float],
    ) -> List[int]:
        """Performs Weighted Reciprocal Rank Fusion."""
        fused_scores = defaultdict(float)
        all_candidates = {cand for scores in per_retriever_scores.values() for cand in scores}

        # Convert scores to ranks (1 = best)
        per_retriever_ranks: Dict[str, Dict[Candidate, int]] = {
            name: self.scores_to_ranks(scores) for name, scores in per_retriever_scores.items()
        }

        for cand in all_candidates:
            val = 0.0
            for name, ranks in per_retriever_ranks.items():
                if cand in ranks:
                    w = float(weights.get(name, 0.0))
                    val += w * (1.0 / (self.rrf_k + ranks[cand]))
            fused_scores[cand] = val

        return sorted(fused_scores, key=fused_scores.get, reverse=True)

    def _weighted_linear_fuse(
        self,
        per_retriever_scores: Dict[str, Dict[Candidate, float]],
        weights: Dict[str, float],
    ) -> List[int]:
        """Performs weighted linear fusion of min-max normalized scores."""
        combined_scores = defaultdict(float)

        for name, scores in per_retriever_scores.items():
            w = float(weights.get(name, 0.0))
            if w <= 0.0:
                continue
            normalized_scores = self.normalize(scores)
            for cand, norm_score in normalized_scores.items():
                combined_scores[cand] += w * norm_score

        return sorted(combined_scores, key=combined_scores.get, reverse=True)

    @staticmethod
    def scores_to_ranks(scores: Dict[Candidate, float]) -> Dict[Candidate, int]:
        """Turns a score dictionary into a 1-based rank dictionary."""
        if not scores:
            return {}
        sorted_candidates = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
        return {idx: rank for rank, idx in enumerate(sorted_candidates, start=1)}

    @staticmethod
    def normalize(scores: Dict[Candidate, float]) -> Dict[Candidate, float]:
        """Maps arbitrary scores to [0,1] using min-max scaling."""
        if not scores:
            return {}
        vals = list(scores.values())
        min_val, max_val = min(vals), max(vals)
        if max_val <= min_val:
            return {i: 0.0 for i in scores}
        return {i: (v - min_val) / (max_val - min_val) for i, v in scores.items()}
