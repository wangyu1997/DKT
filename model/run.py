import math
import time

import torch.optim as optim
from torch.utils.data import DataLoader

from assist_process import AssignmentData
# from my code
from dataset import *
from trace_model import *
from train import train
from evaluate import evaluate

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# load data
data = AssignmentData()

VOCAL_DIM = 2 * data.n_questions + 1
EMBED_DIM = 256
HIDDEN_DIM = 512
OUTPUT_DIM = data.n_questions
N_LAYERS = 4
N_EPOCHS = 20
CLIP = 1
LEARNING_RATE = 30

train_loader = DataLoader(
    TraceDataset(data.train_data, data.n_questions, data.train_longest), batch_size=64, shuffle=True)
test_loader = DataLoader(
    TraceDataset(data.test_data, data.n_questions, data.train_longest), batch_size=64, shuffle=True)

model = TraceModel(VOCAL_DIM, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, device, dropout=0.4).to(device)
model.apply(init_weight)

criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss Function
optimizer = optim.Adam(model.parameters())
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    adjust_learning_rate(optimizer, epoch, LEARNING_RATE)
    start_time = time.time()
    train_loss = train(model, train_loader, optimizer, criterion, device, clip=1)
    valid_loss = evaluate(model, test_loader, criterion, device)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
