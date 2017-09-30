from typing import Callable, List, Dict, Optional, Set, cast, NamedTuple

import numpy as np
from typeguard import check_argument_types

from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.decoders.beam_search_decoder import BeamSearchDecoder
from neuralmonkey.runners.base_runner import (BaseRunner, Executable,
                                              ExecutionResult, NextExecute)
from neuralmonkey.vocabulary import Vocabulary, END_TOKEN

BeamSearchOutput = NamedTuple("BeamSearchOutput",
                              [("step", int),
                               ("scores", List[List[float]]),
                               ("parent_ids", List[List[int]]),
                               ("token_ids", List[List[int]])])

class BeamSearchExecutable(Executable):
    def __init__(self,
                 rank: int,
                 all_encoders: Set[ModelPart],
                 num_sessions: int,
                 decoder: BeamSearchDecoder,
                 vocabulary: Vocabulary,
                 beam_size: int,
                 postprocess: Optional[Callable]) -> None:

        self._rank = rank
        self._num_sessions = num_sessions
        self._all_encoders = all_encoders
        self._decoder = decoder
        self._vocabulary = vocabulary
        self._postprocess = postprocess
        self._beam_size = beam_size
        self._output = BeamSearchOutput(
            step=0,
            scores=np.empty([0, self._beam_size], dtype=float),
            parent_ids=np.empty([0, self._beam_size], dtype=int),
            token_ids=np.empty([0, self._beam_size], dtype=int))

        self._next_feed = [{} for _ in range(self._num_sessions)]
        # If we are ensembling, we run only one step at a time
        if self._num_sessions > 1:
            for fd in self._next_feed:
                fd.update({self._decoder._max_steps: 1})

        self.result = None  # type: ExecutionResult

    def next_to_execute(self) -> NextExecute:
        return (self._all_encoders,
                {'bs_outputs': self._decoder.outputs},
                self._next_feed)

    def collect_results(self, results: List[Dict]) -> None:
        # Recompute logits
        # Only necessary when ensembling models
        prev_logits = []
        for sess_idx, _ in enumerate(results):
            bs_outputs = results[sess_idx]['bs_outputs']
            prev_logits.append(bs_outputs.last_dec_loop_state.prev_logits)

        prev_logits = np.divide(np.sum(prev_logits, 0), self._num_sessions)

        # We are finished
        # The last score is computed
        bs_outputs = results[0]['bs_outputs']
        step = bs_outputs.last_dec_loop_state.step - 1
        self._output = self._output._replace(
            step=self._output.step + step,
            scores=np.append(
                self._output.scores,
                bs_outputs.last_search_step_output.scores[0:step],
                axis=0),
            parent_ids=np.append(
                self._output.parent_ids,
                bs_outputs.last_search_step_output.parent_ids[0:step],
                axis=0),
            token_ids=np.append(
                self._output.token_ids,
                bs_outputs.last_search_step_output.token_ids[0:step],
                axis=0))

        if self._output.step >= self._decoder.max_output_len:
            self.prepare_results(self._output)
            return


        # Prepare the next feed_dict (in ensembles)
        self._next_feed = []
        for sess_idx, _ in enumerate(results):
            bs_outputs = results[sess_idx]['bs_outputs']
            dec_ls = bs_outputs.last_dec_loop_state
            search_state = bs_outputs.last_search_state
            fd = {
                # In ensembles, we execute only one step at a time
                self._decoder._max_steps : 1,
                self._decoder._dec_input_symbol : dec_ls.input_symbol,
                self._decoder._dec_prev_rnn_state : dec_ls.prev_rnn_state,
                self._decoder._dec_prev_rnn_output : dec_ls.prev_rnn_output,
                self._decoder._dec_prev_logits : prev_logits,
                self._decoder._dec_finished : dec_ls.finished,
                self._decoder._search_state : search_state}
            for fetch, value in zip(self._decoder._dec_prev_contexts,
                                    dec_ls.prev_contexts):
                fd.update({fetch : value})
            self._next_feed.append(fd)
        return

    def prepare_results(self, bs_output):
        max_time = bs_output.scores.shape[0] - 1

        output_tokens = []
        hyp_idx = self._rank - 1
        for time in reversed(range(max_time)):
            token_id = bs_output.token_ids[time][hyp_idx]
            token = self._vocabulary.index_to_word[token_id]
            output_tokens.append(token)
            hyp_idx = bs_output.parent_ids[time][hyp_idx]

        output_tokens.reverse()

        before_eos_tokens = []
        for tok in output_tokens:
            if tok == END_TOKEN:
                break
            before_eos_tokens.append(tok)

        if self._postprocess is not None:
            decoded_tokens = self._postprocess([before_eos_tokens])
        else:
            decoded_tokens = [before_eos_tokens]

        self.result = ExecutionResult(
            outputs=decoded_tokens,
            losses=[bs_output.scores[-1][self._rank - 1]],
            scalar_summaries=None,
            histogram_summaries=None,
            image_summaries=None)


class BeamSearchRunner(BaseRunner):
    def __init__(self,
                 output_series: str,
                 decoder: BeamSearchDecoder,
                 rank: int = 1,
                 postprocess: Callable[[List[str]], List[str]] = None) -> None:
        check_argument_types()
        BaseRunner.__init__(self, output_series, decoder)

        if rank < 1 or rank > decoder.beam_size:
            raise ValueError(
                ("Rank of output hypothesis must be between 1 and the beam "
                 "size ({}), was {}.").format(decoder.beam_size, rank))

        self._rank = rank
        self._postprocess = postprocess

    def get_executable(self,
                       compute_losses: bool = False,
                       summaries: bool = True,
                       num_sessions: int = 1) -> BeamSearchExecutable:
        decoder = cast(BeamSearchDecoder, self._decoder)

        return BeamSearchExecutable(
            self._rank, self.all_coders, num_sessions, decoder,
            decoder.vocabulary, decoder.beam_size, self._postprocess)

    @property
    def loss_names(self) -> List[str]:
        return ["beam_search_score"]

    @property
    def decoder_data_id(self) -> Optional[str]:
        return None

def beam_search_runner_range(output_series: str,
                             decoder: BeamSearchDecoder,
                             max_rank: int = None,
                             postprocess: Callable[
                                 [List[str]], List[str]]=None
                            ) -> List[BeamSearchRunner]:
    """A list of beam search runners for a range of ranks from 1 to max_rank.

    This means there is max_rank output series where the n-th series contains
    the n-th best hypothesis from the beam search.

    Args:
        output_series: Prefix of output series.
        decoder: Beam search decoder shared by all runners.
        max_rank: Maximum rank of the hypotheses.
        postprocess: Series-level postprocess applied on output.

    Returns:
        List of beam search runners getting hypotheses with rank from 1 to
        max_rank.
    """
    check_argument_types()

    if max_rank is None:
        max_rank = decoder.beam_size

    if max_rank > decoder.beam_size:
        raise ValueError(
            ("The maximum rank ({}) cannot be "
             "bigger than beam size {}.").format(
                 max_rank, decoder.beam_size))

    return [BeamSearchRunner("{}.rank{:03d}".format(output_series, r),
                             decoder, r, postprocess)
            for r in range(1, max_rank + 1)]
