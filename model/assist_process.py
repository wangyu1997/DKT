from trace_model import *
from synthetic_process import *

class AssignmentData:
    def __init__(self, max_step=0):
        self.max_step = max_step  # manual set longest sequence length
        root_path = '../data/assistments/'
        train_path = root_path + 'builder_train.csv'
        test_path = root_path + 'builder_test.csv'
        print('Loading khan')
        self.questions = {}  # dictionary for questions' id
        self.n_questions = 0  # Calculate unique question numbers (Vocabulary Size)
        train_data, longest, total_answers = self.load_students(train_path)
        self.train_data = train_data
        test_data, test_longest, test_total_answers = self.load_students(test_path)

        self.train_longest = longest
        self.test_longest = test_longest
        total_answers += test_total_answers
        self.test_data = test_data

        print('total answers', total_answers)
        print('longest', longest)
        print('questions ', self.n_questions)

    # parse students info from csv
    def load_students(self, path):
        max_step = self.max_step
        res_data = []
        longest = 0
        total_answers = 0
        f = open(path)
        line = f.readline()
        while line:
            n_step = int(line)
            question_id_str = f.readline().strip().strip('\n').strip(',')
            correct_str = f.readline().strip().strip('\n').strip(',')
            if not n_step or not question_id_str or not correct_str:
                break
            if max_step > 0:
                n_step = max_step
            student = {
                'question_id': np.zeros(n_step),
                'correct': np.zeros(n_step),
                'n_answers': n_step
            }
            # count questions id from 1 (raw+1)
            for idx, raw_id in enumerate(question_id_str.split(',')):
                question_id = int(raw_id.strip()) + 1  # row_id 0 => question_id 1
                if idx > n_step:
                    break
                student['question_id'][idx] = question_id
                if not str(question_id) in self.questions.keys():
                    self.questions[str(question_id)] = True
                    self.n_questions += 1

            for idx, correct_str in enumerate(correct_str.split(',')):
                correct = int(correct_str.strip())  # zero or one
                if idx > n_step:
                    break
                student['correct'][idx] = correct
            if student['n_answers'] >= 2:  # min sequence len > 1
                res_data.append(student)
            longest = max(longest, student['n_answers'])
            total_answers += student['n_answers']
            line = f.readline()
        return res_data, longest, total_answers

