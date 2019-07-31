import pandas as pd
import torch
import numpy as np


#  Function: Get Data
#  Reads a CSV in standard feature format. Each row is a student
#  and each col is the score for the student on the
#  corresponding item.
class SyntheticData:

    def __init__(self, c, q, s, v):
        name = self.get_name(c, q, s, v)
        print('Loading ' + name)
        path = '../data/synthetic/' + name
        raw_data = pd.read_csv(path, header=None)
        total_students = raw_data.shape[0]
        n_questions = raw_data.shape[1]
        n_steps = n_questions - 1
        n_students = int(total_students / 2)  ## test size and train size

        self.n_questions = n_questions
        self.n_students = n_students
        self.n_steps = n_steps

        self.n_test = self.n_students
        self.n_train = self.n_students

        train_data = raw_data[:n_students].values
        test_data = raw_data[n_students:].values

        self.train_data = self.compress_data(train_data)
        self.test_data = self.compress_data(test_data)

        self.train_longest = n_questions
        self.test_longest = n_questions
        self.total_answers = total_students
        self.n_questions = n_questions


        print('total answers', self.total_answers)
        print('longest', self.train_longest)
        print('questions ', self.n_questions)


    def compress_data(self, dataset):
        new_data_set = []
        for row in dataset:
            answers = self.compress_student(row)
            new_data_set.append(answers)
        return new_data_set

    ##  one-hot compressing
    def compress_student(self, answers):
        student = {'question_id': np.zeros(self.n_questions),
                   'correct': np.zeros(self.n_questions),
                   'n_answers': self.n_questions}
        for idx in range(self.n_questions):
            student['question_id'][idx] = idx + 1
            student['correct'][idx] = answers[idx]
        return student

    @staticmethod
    def get_name(c, q, s, v):
        return 'naive_c' + str(c) + \
               '_q' + str(q) + \
               '_s' + str(s) + \
               '_v' + str(v) + '.csv'
