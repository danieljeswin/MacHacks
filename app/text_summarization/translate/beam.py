from __future__ import division
import torch
from app.text_summarization.translate.penalties import PenaltyBuilder


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha,   length_penalty):
        self.alpha = alpha
        penalty_builder = PenaltyBuilder(length_penalty)
        # Term will be subtracted from probability
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)

        return normalized_probs
