import torch.nn as nn


def train(model, iterator, optimizer, criterion, device, clip=1):
    model.train()
    epoch_loss = 0
    for idx, data in enumerate(iterator):
        inputs, questions, answers, seq_lens = data
        optimizer.zero_grad()
        output = model(inputs.to(device), questions.to(device), seq_lens)

        # trg = [batch size,trg sent len]
        # output = [batch size,output sent len]
        output = output.contiguous().view(-1).to(device)
        answers = answers.contiguous().view(-1).to(device)
        loss = criterion(output, answers)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)
