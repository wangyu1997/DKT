import torch.utils.data as data
from utils import *


class TraceDataset(data.Dataset):

    def __init__(self, data_set, n_questions, longest):
        self.n_questions = n_questions
        self.data_set = data_set
        self.longest = longest

    def __getitem__(self, idx):
        # data: seq_len * input_size
        student = self.data_set[idx]
        seq_len = student['n_answers'] - 1  # actual sequence length
        result = load_student_info(student,self.n_questions, True, self.longest)
        return result['x'], result['target_questions'], result['target_answers'], seq_len

    def __len__(self):
        return len(self.data_set)
