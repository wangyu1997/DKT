import numpy as np
import torch.nn as nn


def load_student_info(student, n_questions, padding=False, longest=0):
    answer_num = student['n_answers']
    time_step = answer_num - 1
    max_seq_len = longest - 1
    if time_step > max_seq_len and padding and longest > 0:
        time_step = max_seq_len
    question_id = student['question_id']
    correct = student['correct']
    result = {}
    x = np.zeros(max_seq_len, dtype=np.int)
    q_target_list = np.zeros((max_seq_len, n_questions), dtype=np.float32)
    a_target_list = np.zeros(max_seq_len, dtype=np.float32)
    for t in range(time_step):
        q_t = question_id[t]
        a_t = correct[t]
        x[t] = q_t * 2 - a_t  # generate unique id (q_t)*2-a_t
        q_target_index = int(question_id[t + 1]) - 1
        q_target_list[t][q_target_index] = 1
        a_target = correct[t + 1]
        a_target_list[t] = a_target
    result['x'] = np.array(x)
    result['target_questions'] = np.array(q_target_list)
    result['target_answers'] = np.array(a_target_list)
    return result


def init_weight(model):
    init_range = 0.05
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param, -init_range, init_range)
        else:
            nn.init.zeros_(param)


def epoch_time(s_time, e_time):
    elapsed_time = e_time - s_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 5))
    if lr < 0.01:
        lr = 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
