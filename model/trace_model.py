from torch.nn import init
from lstm_encoder import *

class TraceModel(nn.Module):

    def __init__(self, vocal_dim, embed_dim, hidden_dim, output_dim, n_layers, device, dropout=0.5):
        super(TraceModel, self).__init__()

        self.vocal_dim = vocal_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.drop_rate = dropout
        self.device = device

        self._build_model()

    def _build_model(self):
        self._init_h = nn.Parameter(
            torch.Tensor(self.n_layers, self.hidden_dim)).to(self.device)
        self._init_c = nn.Parameter(
            torch.Tensor(self.n_layers, self.hidden_dim)).to(self.device)
        init.xavier_normal_(self._init_h)
        init.xavier_normal_(self._init_c)

        self._lstm = nn.LSTM(
            input_size=self.embed_dim,
            num_layers=self.n_layers,
            hidden_size=self.hidden_dim,
            dropout=self.drop_rate,
            batch_first=True
        )

        self.dropout = nn.Dropout(self.drop_rate)
        self.embedding = nn.Embedding(self.vocal_dim, self.embed_dim)
        self.full_connect = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_, trg, in_lens=None):
        # lstm encoder
        # input size [batch size, max_sequence_len]
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, _ = lstm_encoder(
            input_, self._lstm, in_lens, init_states, self.embedding)
        # full connect
        y_t = self.sigmoid(self.full_connect(self.dropout(lstm_out)))
        # y_t size [batch size, max_sequence_len, class_num]
        # mask [batch size, max_sequence_len]
        mask = self.make_mask(input_)
        mask_expand = mask.unsqueeze(2).expand(*y_t.size())
        y_masked = y_t.masked_fill(mask=mask_expand, value=0)
        # y_masked size [batch size,max_sequence_len,classes]

        # trg size [batch size, max_sequence_len, class] => q_t+1 one_hot
        output = y_masked.mul(trg).sum(dim=2, keepdim=True).unsqueeze(2)
        # output size [batch size, max_sequence_len]
        return output

    @staticmethod
    def make_mask(sequence):
        mask = sequence.eq(0)
        return mask
