import torch.nn as nn
import torch


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(iterator):
            inputs, questions, answers, seq_lens = data
            output = model(inputs.to(device), questions.to(device), seq_lens)

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            output = output.contiguous().view(-1).to(device)
            answers = answers.contiguous().view(-1).to(device)
            loss = criterion(output, answers)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
