# pylint: disable=too-many-lines
from typing import List, Callable, Optional, Any, NamedTuple, Union

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.decoders.sequence_decoder import SequenceDecoder, LoopState
from neuralmonkey.attention.base_attention import BaseAttention
from neuralmonkey.vocabulary import (Vocabulary, END_TOKEN_INDEX,
                                     PAD_TOKEN_INDEX)
from neuralmonkey.model.sequence import EmbeddedSequence
from neuralmonkey.model.stateful import (TemporalStatefulWithOutput,
                                         SpatialStatefulWithOutput)
from neuralmonkey.logging import log, warn
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell
from neuralmonkey.nn.utils import dropout
from neuralmonkey.decoders.encoder_projection import (
    linear_encoder_projection, concat_encoder_projection, empty_initial_state)
from neuralmonkey.decoders.output_projection import (OutputProjectionSpec,
                                                     nonlinear_output)
from neuralmonkey.decorators import tensor


RNN_CELL_TYPES = {
    "GRU": OrthoGRUCell,
    "LSTM": tf.contrib.rnn.LSTMCell
}

RNNLoopState = NamedTuple(
    "RNNLoopState",
    [("input_symbol", tf.Tensor),  # batch of ints to vocab
     ("train_inputs", Optional[tf.Tensor]),
     ("prev_rnn_state", tf.Tensor),
     ("prev_rnn_output", tf.Tensor),
     ("prev_logits", tf.Tensor),
     ("prev_contexts", List[tf.Tensor]),
     ("attention_loop_states", List[Any])])  # see att docs


# pylint: disable=too-many-instance-attributes
class Decoder(SequenceDecoder):
    """A class that manages parts of the computation graph that are
    used for the decoding.
    """

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments,too-many-branches,too-many-statements
    def __init__(self,
                 # TODO only stateful, attention will need temporal or spat.
                 encoders: List[Union[TemporalStatefulWithOutput,
                                      SpatialStatefulWithOutput]],
                 vocabulary: Vocabulary,
                 data_id: str,
                 name: str,
                 max_output_len: int,
                 dropout_keep_prob: float = 1.0,
                 rnn_size: int = None,
                 embedding_size: int = None,
                 output_projection: OutputProjectionSpec = None,
                 encoder_projection: Callable[
                     [tf.Tensor, Optional[int], Optional[List[Any]]],
                     tf.Tensor]=None,
                 attentions: List[BaseAttention] = None,
                 embeddings_source: EmbeddedSequence = None,
                 attention_on_input: bool = True,
                 rnn_cell: str = 'GRU',
                 conditional_gru: bool = False,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
        """Create a refactored version of monster decoder.

        Arguments:
            encoders: Input encoders of the decoder
            vocabulary: Target vocabulary
            data_id: Target data series
            name: Name of the decoder. Should be unique accross all Neural
                Monkey objects
            max_output_len: Maximum length of an output sequence
            dropout_keep_prob: Probability of keeping a value during dropout

        Keyword arguments:
            rnn_size: Size of the decoder hidden state, if None set
                according to encoders.
            embedding_size: Size of embedding vectors for target words
            output_projection: How to generate distribution over vocabulary
                from decoder_outputs
            encoder_projection: How to construct initial state from encoders
            attention: The attention object to use. Optional.
            embeddings_source: Embedded sequence to take embeddings from
            rnn_cell: RNN Cell used by the decoder (GRU or LSTM)
            conditional_gru: Flag whether to use the Conditional GRU
                architecture
            attention_on_input: Flag whether attention from previous decoding
                step should be combined with the input in the next step.
        """
        check_argument_types()
        SequenceDecoder.__init__(
            self,
            name=name,
            vocabulary=vocabulary,
            data_id=data_id,
            max_output_len=max_output_len,
            dropout_keep_prob=dropout_keep_prob,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint)

        self.encoders = encoders
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.output_projection_spec = output_projection
        self.encoder_projection = encoder_projection
        self.attentions = attentions
        self.embeddings_source = embeddings_source
        self._conditional_gru = conditional_gru
        self._attention_on_input = attention_on_input
        self._rnn_cell_str = rnn_cell

        if self.attentions is None:
            self.attentions = []

        if self.embedding_size is None and self.embeddings_source is None:
            raise ValueError("You must specify either embedding size or the "
                             "embedded sequence from which to reuse the "
                             "embeddings (e.g. set either 'embedding_size' or "
                             " 'embeddings_source' parameter)")

        if self.embeddings_source is not None:
            if self.embedding_size is not None:
                warn("Overriding the embedding_size parameter with the"
                     " size of the reused embeddings from the encoder.")

            self.embedding_size = (
                self.embeddings_source.embedding_matrix.get_shape()[1].value)

        if self.encoder_projection is None:
            if not self.encoders:
                log("No direct encoder input. Using empty initial state")
                self.encoder_projection = empty_initial_state
            elif rnn_size is None:
                log("No rnn_size or encoder_projection: Using concatenation of"
                    " encoded states")
                self.encoder_projection = concat_encoder_projection
                self.rnn_size = sum(e.output.get_shape()[1].value
                                    for e in encoders)
            else:
                log("Using linear projection of encoders as the initial state")
                self.encoder_projection = linear_encoder_projection(
                    self.dropout_keep_prob)

        assert self.rnn_size is not None

        if self._rnn_cell_str not in RNN_CELL_TYPES:
            raise ValueError("RNN cell must be a either 'GRU' or 'LSTM'")

        if self.output_projection_spec is None:
            log("No output projection specified - using tanh projection")
            self.output_projection = nonlinear_output(
                self.rnn_size, tf.tanh)[0]
            self.output_projection_size = self.rnn_size
        elif isinstance(self.output_projection_spec, tuple):
            (self.output_projection,
             self.output_projection_size) = tuple(self.output_projection_spec)
        else:
            self.output_projection = self.output_projection_spec
            self.output_projection_size = self.rnn_size

        if self._attention_on_input:
            self.input_projection = self.input_plus_attention
        else:
            self.input_projection = self.embed_input_symbol

        with self.use_scope():
            with tf.variable_scope("attention_decoder") as self.step_scope:
                pass

        # TODO when it is possible, remove the printing of the cost var
        log("Decoder initalized. Cost var: {}".format(str(self.cost)))
        log("Runtime logits tensor: {}".format(str(self.runtime_logits)))
    # pylint: enable=too-many-arguments,too-many-branches,too-many-statements

    @tensor
    def initial_state(self) -> tf.Tensor:
        """The part of the computation graph that computes
        the initial state of the decoder.
        """
        with tf.variable_scope("initial_state"):
            initial_state = dropout(
                self.encoder_projection(self.train_mode,
                                        self.rnn_size,
                                        self.encoders),
                self.dropout_keep_prob,
                self.train_mode)

            # pylint: disable=no-member
            # Pylint keeps complaining about initial shape being a tuple,
            # but it is a tensor!!!
            init_state_shape = initial_state.get_shape()
            # pylint: enable=no-member

            # Broadcast the initial state to the whole batch if needed
            if len(init_state_shape) == 1:
                assert init_state_shape[0].value == self.rnn_size
                tiles = tf.tile(initial_state,
                                tf.expand_dims(self.batch_size, 0))
                initial_state = tf.reshape(tiles, [-1, self.rnn_size])

        return initial_state

    @tensor
    def embedding_matrix(self) -> tf.Variable:
        """Variables and operations for embedding of input words

        If we are reusing word embeddings, this function takes the embedding
        matrix from the first encoder
        """
        if self.embeddings_source is not None:
            return self.embeddings_source.embedding_matrix

        # TODO better initialization
        return tf.get_variable(
            "word_embeddings", [len(self.vocabulary), self.embedding_size],
            initializer=tf.random_uniform_initializer(-0.5, 0.5))

    @property
    def output_dimension(self) -> Union[int, tf.Tensor]:
        return self.output_projection_size

    def _get_rnn_cell(self) -> tf.contrib.rnn.RNNCell:
        return RNN_CELL_TYPES[self._rnn_cell_str](self.rnn_size)

    def _get_conditional_gru_cell(self) -> tf.contrib.rnn.GRUCell:
        return tf.contrib.rnn.GRUCell(self.rnn_size)

    def embed_input_symbol(self, *args) -> tf.Tensor:
        loop_state = LoopState(*args)
        rnn_ls = loop_state.dec_ls

        embedded_input = tf.nn.embedding_lookup(
            self.embedding_matrix, rnn_ls.input_symbol)

        return dropout(embedded_input, self.dropout_keep_prob, self.train_mode)

    def input_plus_attention(self, *args) -> tf.Tensor:
        """Merge input and previous attentions into one vector of the
         right size.
        """
        loop_state = LoopState(*args)
        rnn_ls = loop_state.dec_ls

        embedded_input = self.embed_input_symbol(*loop_state)
        emb_with_ctx = tf.concat(
            [embedded_input] + rnn_ls.prev_contexts, 1)

        return tf.layers.dense(emb_with_ctx, self.embedding_size)

    def get_body(self,
                 train_mode: bool,
                 sample: bool = False) -> Callable:
        # pylint: disable=too-many-branches
        def body(*args) -> LoopState:
            loop_state = LoopState(*args)
            rnn_ls = loop_state.dec_ls
            step = loop_state.step

            with tf.variable_scope(self.step_scope):
                # Compute the input to the RNN
                rnn_input = self.input_projection(*loop_state)

                # Run the RNN.
                cell = self._get_rnn_cell()
                if self._rnn_cell_str == 'GRU':
                    cell_output, state = cell(rnn_input,
                                              rnn_ls.prev_rnn_output)
                    next_state = state
                    attns = [
                        a.attention(cell_output, rnn_ls.prev_rnn_output,
                                    rnn_input, att_loop_state, loop_state.step)
                        for a, att_loop_state in zip(
                            self.attentions,
                            rnn_ls.attention_loop_states)]
                    if self.attentions:
                        contexts, att_loop_states = zip(*attns)
                    else:
                        contexts, att_loop_states = [], []

                    if self._conditional_gru:
                        cell_cond = self._get_conditional_gru_cell()
                        cond_input = tf.concat(contexts, -1)
                        cell_output, state = cell_cond(cond_input, state,
                                                       scope="cond_gru_2_cell")
                elif self._rnn_cell_str == 'LSTM':
                    prev_state = tf.contrib.rnn.LSTMStateTuple(
                        rnn_ls.prev_rnn_state, rnn_ls.prev_rnn_output)
                    cell_output, state = cell(rnn_input, prev_state)
                    next_state = state.c
                    attns = [
                        a.attention(cell_output, rnn_ls.prev_rnn_output,
                                    rnn_input, att_loop_state, loop_state.step)
                        for a, att_loop_state in zip(
                            self.attentions,
                            rnn_ls.attention_loop_states)]
                    if self.attentions:
                        contexts, att_loop_states = zip(*attns)
                    else:
                        contexts, att_loop_states = [], []
                else:
                    raise ValueError("Unknown RNN cell.")

                with tf.name_scope("rnn_output_projection"):
                    embedded_input = tf.nn.embedding_lookup(
                        self.embedding_matrix, rnn_ls.input_symbol)

                    output = self.output_projection(
                        cell_output, embedded_input, list(contexts),
                        self.train_mode)

                logits = self.get_logits(output)

            self.step_scope.reuse_variables()

            if sample:
                next_symbols = tf.multinomial(logits, num_samples=1)
            elif train_mode:
                next_symbols = rnn_ls.train_inputs[step]
            else:
                next_symbols = tf.to_int32(tf.argmax(logits, axis=1))
                int_unfinished_mask = tf.to_int32(
                    tf.logical_not(loop_state.finished))

                # Note this works only when PAD_TOKEN_INDEX is 0. Otherwise
                # this have to be rewritten
                assert PAD_TOKEN_INDEX == 0
                next_symbols = next_symbols * int_unfinished_mask

            has_just_finished = tf.equal(next_symbols, END_TOKEN_INDEX)
            has_finished = tf.logical_or(loop_state.finished,
                                         has_just_finished)
            not_finished = tf.logical_not(has_finished)

            new_rnn_ls = RNNLoopState(
                input_symbol=next_symbols,
                train_inputs=rnn_ls.train_inputs,
                prev_rnn_state=next_state,
                prev_rnn_output=cell_output,
                prev_logits=logits,
                prev_contexts=list(contexts),
                attention_loop_states=list(att_loop_states))

            new_loop_state = LoopState(
                step=step + 1,
                decoder_outputs=loop_state.decoder_outputs.write(
                    step + 1, cell_output),
                logits=loop_state.logits.write(step, logits),
                finished=has_finished,
                mask=loop_state.mask.write(step, not_finished),
                dec_ls=new_rnn_ls)

            return new_loop_state
        # pylint: enable=too-many-branches

        return body

    def get_initial_loop_state(self) -> LoopState:
        rnn_output_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True,
                                       size=0, name="decoder_outputs")
        rnn_output_ta = rnn_output_ta.write(0, self.initial_state)

        logit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True,
                                  size=0, name="logits")

        contexts = [tf.zeros([self.batch_size, a.context_vector_size])
                    for a in self.attentions]

        mask_ta = tf.TensorArray(dtype=tf.bool, dynamic_size=True,
                                 size=0, name="mask")

        attn_loop_states = [a.initial_loop_state()
                            for a in self.attentions if a is not None]

        rnn_loop_state = RNNLoopState(
            input_symbol=self.go_symbols,
            train_inputs=self.train_inputs,
            prev_rnn_state=self.initial_state,
            prev_rnn_output=self.initial_state,
            prev_logits=tf.zeros([self.batch_size, len(self.vocabulary)]),
            prev_contexts=contexts,
            attention_loop_states=attn_loop_states)

        return LoopState(
            step=0,
            decoder_outputs=rnn_output_ta,
            logits=logit_ta,
            mask=mask_ta,
            finished=tf.zeros([self.batch_size], dtype=tf.bool),
            dec_ls=rnn_loop_state)

    def finalize_loop(self, final_loop_state: LoopState,
                      train_mode: bool) -> None:
        rnn_ls = final_loop_state.dec_ls

        for att_state, attn_obj in zip(
                rnn_ls.attention_loop_states, self.attentions):

            att_history_key = "{}_{}".format(
                self.name, "train" if train_mode else "run")

            attn_obj.finalize_loop(att_history_key, att_state)

            if not train_mode:
                attn_obj.visualize_attention(att_history_key)
