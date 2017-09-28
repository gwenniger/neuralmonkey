"""Implementation of the decoder of the Transformer model as described in
Vaswani et al. (2017).

See arxiv.org/abs/1706.03762
"""
from typing import Callable, Set, List, Optional

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.attention.scaled_dot_product import MultiHeadAttention
from neuralmonkey.decorators import tensor
from neuralmonkey.decoders.sequence_decoder import SequenceDecoder, LoopState
from neuralmonkey.encoders.transformer import (TransformerLayer,
                                               TransformerEncoder)
from neuralmonkey.logging import log
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.nn.projection import linear, nonlinear
from neuralmonkey.nn.utils import dropout
from neuralmonkey.vocabulary import Vocabulary


class TransformerDecoder(SequenceDecoder):

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoder: TransformerEncoder,
                 vocabulary: Vocabulary,
                 data_id: str,
                 # TODO infer the default for these three from the encoder
                 ff_hidden_size: int,
                 n_heads_self: int,
                 n_heads_enc: int,
                 depth: int,
                 max_output_len: int = None,
                 dropout_keep_prob: float = 1.0,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None) -> None:
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

        self.encoder = encoder
        self.ff_hidden_size = ff_hidden_size
        self.n_heads_self = n_heads_self
        self.n_heads_enc = n_heads_enc
        self.depth = depth

        self.dimension = self.encoder.dimension

        self.self_attentions = [None for _ in range(self.depth)] \
            # type: List[Optional[MultiHeadAttention]]
        self.inter_attentions = [None for _ in range(self.depth)] \
            # type: List[Optional[MultiHeadAttention]]

        log("Decoder cost op: {}".format(self.cost))
    # pylint: enable=too-many-arguments

    @property
    def output_dimension(self) -> int:
        return self.dimension

    @tensor
    def embedding_matrix(self) -> tf.Variable:
        # TODO better initialization
        return tf.get_variable(
            "word_embeddings", [len(self.vocabulary), self.dimension],
            initializer=tf.random_uniform_initializer(-0.5, 0.5))

    def _position_signal(self, length: tf.Tensor) -> tf.Tensor:
        # TODO DUPLICATE CODE, THE SAME FUNCTION IS IN TRANSFORMER ENCODER

        # code simplified and copied from github.com/tensorflow/tensor2tensor

        # TODO write this down on a piece of paper and understand the code and
        # compare it to the paper
        positions = tf.to_float(tf.range(length))

        num_timescales = self.dimension // 2
        log_timescale_increment = 4 / (tf.to_float(num_timescales) - 1)

        inv_timescales = tf.exp(tf.to_float(tf.range(num_timescales))
                                * -log_timescale_increment)

        scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(
            inv_timescales, 0)

        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(self.dimension, 2)]])
        signal = tf.reshape(signal, [1, length, self.dimension])

        return signal

    @tensor
    def decoder_inputs(self) -> tf.Tensor:

        shifted = tf.concat([tf.expand_dims(self.go_symbols, 1),
                             tf.transpose(self.train_inputs)], 1)

        return tf.to_float(shifted) + self._position_signal(tf.shape(self.train_inputs)[0])

    def layer(self, level: int) -> TransformerLayer:

        ### MASKOVANI SELF-ATTENTIONU BEHEM RUNTIME
        ### BEHEM TRAIN JE TAKY MASKOVANEJ

        ### MASKUJOU SE KLICE I HODNOTY?

        ### TAKY SE TO MUSI POSUNOUT O JEDNO DOPRAVA ABYCHOM MOHLI DAT <S>

        if level == 0:
            return TransformerLayer(self.decoder_inputs,
                                    tf.transpose(self.train_mask))  # WAT?

        prev_layer = self.layer(level - 1)

        # Compute the outputs of this layer
        s_ckp = "dec_self_att_{}_{}".format(
            level, self._save_checkpoint) if self._save_checkpoint else None
        l_ckp = "dec_self_att_{}_{}".format(
            level, self._load_checkpoint) if self._load_checkpoint else None

        att = MultiHeadAttention(name="mask_self_att_{}".format(level),
                                 n_heads=self.n_heads_self,
                                 keys_encoder=prev_layer,
                                 values_encoder=prev_layer,
                                 dropout_keep_prob=self.dropout_keep_prob,
                                 save_checkpoint=s_ckp,
                                 load_checkpoint=l_ckp)

        # TODO generalize att work with 3D queries as default
        with tf.variable_scope("dec_self_att_level_{}".format(level)):
            self_att_result = att.attention_3d(
                prev_layer.temporal_states, masked=True)
            self.self_attentions[level - 1] = att

        inter_attention_query = tf.contrib.layers.layer_norm(
            self_att_result + prev_layer.temporal_states)

        att_i = MultiHeadAttention(name="inter_att_{}".format(level),
                                   n_heads=self.n_heads_enc,
                                   keys_encoder=self.encoder,
                                   values_encoder=self.encoder,
                                   dropout_keep_prob=self.dropout_keep_prob,
                                   save_checkpoint=s_ckp,
                                   load_checkpoint=l_ckp)

        # TODO generalize att work with 3D queries as default
        with tf.variable_scope("dec_inter_att_level_{}".format(level)):
            inter_att_result = att.attention_3d(inter_attention_query)
            self.inter_attentions[level - 1] = att_i

        ff_input = tf.contrib.layers.layer_norm(
            inter_att_result + inter_attention_query)

        ff_hidden = nonlinear(ff_input, self.ff_hidden_size,
                              activation=tf.nn.relu,
                              scope="ff_hidden_{}".format(level))

        ff_output = linear(ff_hidden, self.dimension,
                           scope="ff_out_{}".format(level))

        output_states = tf.contrib.layers.layer_norm(ff_output + ff_input)
        return TransformerLayer(states=output_states, mask=self.train_mask)

    @tensor
    def train_logits(self) -> tf.Tensor:
        last_layer = self.layer(self.depth)

        temporal_states = dropout(last_layer.temporal_states,
                                  self.dropout_keep_prob, self.train_mode)

        # matmul with output matrix
        # t_states shape: (batch, time, channels)
        # dec_w shape: (channels, vocab)

        logits = tf.nn.conv1d(
            temporal_states, tf.expand_dims(self.decoding_w, 0), 1, "SAME")

        return logits + tf.expand_dims(tf.expand_dims(self.decoding_b, 0), 0)

    def get_initial_loop_state(self) -> LoopState:
        raise NotImplementedError("Abstract method")

    def get_body(self, train_mode: bool, sample: bool = False) -> Callable:
        assert not train_mode

        def body(*args) -> LoopState:
            loop_state = LoopState(*args)
            step = loop_state.step

            # INFERENCE
            # mam novy slovo a udelam krok transformera

            # TOHLE JE TO SAMY JAKO POUSTET HO NA CELY, JEN POKAZDY S JINOU
            # MAX DYLKOU A MASKOU

    def get_dependencies(self) -> Set[ModelPart]:
        assert all(self.self_attentions)
        assert all(self.inter_attentions)

        dependencies = self.self_attentions + self.inter_attentions

        return set(dependencies)
