from sklearn.metrics import accuracy_score
import copy

def scoring(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
    
    
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    accumulation_steps = 3

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)
        loss = loss / accumulation_steps
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if((i+1)%accumulation_steps)==0:
            optimizer.step()        # 反向传播，更新网络参数
            optimizer.zero_grad()   # 清空梯度
            
            
def validate(val_loader, model):
    model.eval()
    
    outputs, targets = [], []
    with torch.no_grad():
        for i, (input, labels) in enumerate(val_loader):
            input = input.cuda()
            labels = labels.cuda()
            output = model(input)
            outputs.append(output)
            targets.append(labels)
    score = scoring(
                torch.cat(targets, dim=0).cpu(),
                torch.cat(outputs, dim=0).cpu().argmax(1)
            )
    return score
