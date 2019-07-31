import torch
import torch.nn as nn


def lstm_encoder(sequence, lstm,
                 seq_lens=None, init_states=None, embedding=None):
    """ functional LSTM encoder (sequence is [batch, seq_len, dim],
    lstm should be rolled lstm)"""
    # transpose batch tensor to fit lstm format
    # sequence size [batch size,max_seq_len]
    batch_size = sequence.size(0)
    max_seq_len = sequence.size(1)
    batch_first = lstm.batch_first

    if not batch_first:  # embedding and transpose input sequence tensor
        sequence = sequence.transpose(0, 1)

    # emb_sequence size [batch size,max_seq_len,emb_dim]
    emb_sequence = (embedding(sequence) if embedding is not None
                    else sequence)
    # reorder batch tensor along batch dim
    if not seq_lens is None:  # reorder input sequence tensor along batch dim
        # (max_sen_len, batch_size, lstm_input_size) 按照batch_size维度，根据文本实际长度（句子数量）降序排列
        assert batch_size == len(seq_lens)
        sort_ind = sorted(range(len(seq_lens)),
                          key=lambda i: seq_lens[i], reverse=True)  # 确定排序索引
        seq_lens = [seq_lens[i] for i in sort_ind]  # 根据排序索引 对序列真实长度进行排序
        sequence = reorder_sequence(emb_sequence, sort_ind,
                                    lstm.batch_first)  # 根据排序索引对tensor batch dim进行排序

    # init hidden state and cell state for lstm
    if init_states is None:  # 初始化lstm中的hidden state 和 cell state
        device = sequence.device
        init_states = init_lstm_states(lstm, batch_size, device)
    else:
        init_states = (init_states[0].contiguous(),
                       init_states[1].contiguous())

    if not seq_lens is None:  # Encode & Reorder Back
        packed_seq = nn.utils.rnn.pack_padded_sequence(emb_sequence,  # 压缩lstm输入序列，保留输入序列更多有效序列
                                                       seq_lens,
                                                       batch_first=batch_first)  # https://www.cnblogs.com/sbj123456789/p/9834018.html
        packed_out, final_states = lstm(packed_seq.to(init_states[0].dtype), init_states)  # encode
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=batch_first,
                                                       total_length=max_seq_len)
        # (max_sent_len, batch_size, emb_dim)

        sort_ind = sorted(range(len(seq_lens)),
                          key=lambda i: seq_lens[i], reverse=True)  # 确定排序索引
        back_map = {ind: i for i, ind in enumerate(sort_ind)}  # 结构为{之前索引： 当前索引}， 将编码之后的结果按照索引对应回输入索引
        reorder_ind = [back_map[i] for i in range(len(seq_lens))]  # 生成逆排序索引，对应于sort_ind
        lstm_out = reorder_sequence(lstm_out, reorder_ind,
                                    batch_first)  # 根据逆排序索引对tensor batch dim进行排序 (max_sent_len, batch_size, lstm_size)
        # final_states = reorder_lstm_states(final_states, reorder_ind)
    else:
        lstm_out, final_states = lstm(sequence, init_states)

    # transpose
    return lstm_out, final_states  # (seq_len, batch, embedding)  (hidden_layer* direction_num, batch, hidden_size)


def reorder_sequence(sequence, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence.size()[batch_dim]  # 确认排序索引和该batch_size长度是否一致
    order = torch.LongTensor(order).to(sequence.device)
    sorted_ = sequence.index_select(index=order,
                                    dim=batch_dim)  # Returns a new tensor which indexes the sequence_emb along dimension batch_dim using order

    return sorted_


def init_lstm_states(lstm, batch_size, device):
    n_layer = lstm.num_layers * (2 if lstm.bidirectional else 1)
    n_hidden = lstm.hidden_size
    states = (torch.zeros(n_layer, batch_size, n_hidden).to(device),
              torch.zeros(n_layer, batch_size, n_hidden).to(device))
    return states
