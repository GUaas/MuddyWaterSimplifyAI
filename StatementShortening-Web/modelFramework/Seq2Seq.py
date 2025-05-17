import paddle

class Encoder(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(Encoder, self).__init__()

        self.embedding = paddle.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = paddle.nn.LSTM(input_size=embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=0.2 if num_layers > 1 else 0)

    def forward(self, src, src_length):
        inputs = self.embedding(src)  # [batch_size,time_steps,embedding_dim]
        encoder_out, encoder_state = self.lstm(inputs,
                                               sequence_length=src_length)
        return encoder_out, encoder_state


class AttentionLayer(paddle.nn.Layer):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attn1 = paddle.nn.Linear(hidden_size, hidden_size)
        self.attn2 = paddle.nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(self, decoder_hidden_h, encoder_output, encoder_padding_mask):
        encoder_output = self.attn1(encoder_output)

        a = paddle.unsqueeze(decoder_hidden_h, [1])
        attn_scores = paddle.matmul(a, encoder_output, transpose_y=True)

        if encoder_padding_mask is not None:
            attn_scores = paddle.add(attn_scores, encoder_padding_mask)

        attn_scores = paddle.nn.functional.softmax(attn_scores)
        attn_out = paddle.squeeze(paddle.matmul(attn_scores, encoder_output), [1])
        attn_out = paddle.concat([attn_out, decoder_hidden_h], 1)
        attn_out = self.attn2(attn_out)

        return attn_out


class DecoderCell(paddle.nn.RNNCellBase):
    def __init__(self, num_layers, embedding_dim, hidden_size):
        super(DecoderCell, self).__init__()

        self.dropout = paddle.nn.Dropout(0.2)
        self.lstmcells = paddle.nn.LayerList([paddle.nn.LSTMCell(
            input_size=embedding_dim + hidden_size if i == 0 else hidden_size,
            hidden_size=hidden_size
        ) for i in range(num_layers)])

        self.attention = AttentionLayer(hidden_size)

    def forward(self, decoder_input, decoder_initial_states, encoder_out, encoder_padding_mask=None):
        encoder_final_states, decoder_init_states = decoder_initial_states
        new_lstm_states = []
        inputs = paddle.concat([decoder_input, decoder_init_states], 1)

        for i, lstm_cell in enumerate(self.lstmcells):
            state_h, new_lstm_state = lstm_cell(inputs, encoder_final_states[i])
            inputs = self.dropout(state_h)
            new_lstm_states.append(new_lstm_state)

        state_h = self.attention(inputs, encoder_out, encoder_padding_mask)

        return state_h, [new_lstm_states, state_h]


class Decoder(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = paddle.nn.Embedding(vocab_size, embedding_dim)
        self.lstm_attention = paddle.nn.RNN(DecoderCell(num_layers, embedding_dim, hidden_size))
        self.fianl = paddle.nn.Linear(hidden_size, vocab_size)

    def forward(self, trg, decoder_initial_states, encoder_output, encoder_padding_mask):
        inputs = self.embedding(trg)
        decoder_out, _ = self.lstm_attention(inputs,
                                             initial_states=decoder_initial_states,
                                             encoder_out=encoder_output,
                                             encoder_padding_mask=encoder_padding_mask)
        predict = self.fianl(decoder_out)

        return predict


class Seq2Seq(paddle.nn.Layer):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, eos_id):
        super(Seq2Seq, self).__init__()

        self.hidden_size = hidden_size
        self.eos_id = eos_id
        self.num_layers = num_layers
        self.INF = 1e9

        self.encoder = Encoder(vocab_size, embedding_dim, hidden_size, num_layers)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size, num_layers)

    def forward(self, src, src_length, trg):
        encoder_output, encoder_final_state = self.encoder(src, src_length)
        encoder_final_states = [(encoder_final_state[0][i], encoder_final_state[1][i]) for i in range(self.num_layers)]

        decoder_initial_states = [encoder_final_states,
                                  self.decoder.lstm_attention.cell.get_initial_states(batch_ref=encoder_output,
                                                                                      shape=[self.hidden_size])]

        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())
        encoder_mask = (src_mask - 1) * self.INF
        encoder_padding_mask = paddle.unsqueeze(encoder_mask, [1])

        predict = self.decoder(trg, decoder_initial_states, encoder_output, encoder_padding_mask)

        return predict


class Seq2SeqInfer(Seq2Seq):
    def __init__(self, word_size, embedding_dim, hidden_size, num_layers, bos_id, eos_id, beam_size,
                 max_out_len=None):
        self.bos_id = bos_id
        self.beam_size = beam_size
        self.max_out_len = max_out_len
        self.num_layers = num_layers

        super(Seq2SeqInfer, self).__init__(word_size, embedding_dim, hidden_size, num_layers, eos_id)

        self.beam_search_decoder = paddle.nn.BeamSearchDecoder(
            self.decoder.lstm_attention.cell,
            start_token=bos_id,
            end_token=eos_id,
            beam_size=beam_size,
            embedding_fn=self.decoder.embedding,
            output_fn=self.decoder.fianl)

    def forward(self, src, src_length):
        encoder_output, encoder_states = self.encoder(src, src_length)

        encoder_final_state = [(encoder_states[0][i], encoder_states[1][i]) for i in range(self.num_layers)]

        decoder_initial_states = [encoder_final_state,
                                  self.decoder.lstm_attention.cell.get_initial_states(batch_ref=encoder_output,
                                                                                      shape=[self.hidden_size])]

        src_mask = (src != self.eos_id).astype(paddle.get_default_dtype())
        encoder_padding_mask = (src_mask - 1.0) * self.INF
        encoder_padding_mask = paddle.unsqueeze(encoder_padding_mask, [1])

        encoder_out = paddle.nn.BeamSearchDecoder.tile_beam_merge_with_batch(encoder_output, self.beam_size)
        encoder_padding_mask = paddle.nn.BeamSearchDecoder.tile_beam_merge_with_batch(encoder_padding_mask,
                                                                                      self.beam_size)

        seq_output, _ = paddle.nn.dynamic_decode(decoder=self.beam_search_decoder,
                                                 inits=decoder_initial_states,
                                                 max_step_num=self.max_out_len,
                                                 encoder_out=encoder_output,
                                                 encoder_padding_mask=encoder_padding_mask)

        return seq_output