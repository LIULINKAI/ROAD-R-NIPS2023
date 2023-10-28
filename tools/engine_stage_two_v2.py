from tqdm import tqdm

import torch

def multi_label_acc(output, target, pos_thresh=0.5):
    output1 = output.detach().cpu().numpy()
    target1 = target.detach().cpu().numpy()
    output1[output1 > pos_thresh] = 1
    output1[output1 <= pos_thresh] = 0
    acc = 0
    seq_len = output1.shape[1]
    for i in range(output1.shape[0]):
        for t in range(seq_len):
            if (output1[i][t] == target1[i][t]).sum() == output1[i][t].shape[0]:
                acc += 1
    return acc/seq_len

def train(args, model, train_loader, optimizer, criterion, epoch, writer):
    model.train()
    
    train_loss, train_acc = 0, 0
    tqdm_iter = tqdm(train_loader, desc="Epoch: {}/{} ({}%) |Training loss: NaN".format(
        epoch, args.epoch, int(epoch/args.epoch)), leave=False)
    for batch_idx, (data, label, loc, agent_id) in enumerate(tqdm_iter):
        data, target = data.to(args.device), label.to(args.device)
        loc = loc.to(torch.float32).to(args.device)
        agent_id = agent_id.to(args.device)
        output = model(data, loc, agent_id)
        # loss = criterion(output, target)
        output = torch.sigmoid(output)
        num_pos = max(1.0, (target > 0).sum().item())
        loss = criterion(output, target, num_pos, args.alpha, args.gamma)
        acc = multi_label_acc(output=output, target=target, pos_thresh=args.acc_pos_thresh)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # acc = (output.argmax(dim=1) == target.argmax(dim=1)).float().sum().item()

        train_loss += loss.cpu().item()
        train_acc += acc

        tqdm_iter.set_description("Epoch: {}/{} ({}%) |Training loss: {:.6f} |Training Acc: {:.6f}".format(
            epoch, args.epoch, int(epoch/args.epoch), round(loss.item(), 6), round(acc / args.batch_size, 6)))
        
        if epoch == 1:
            writer.add_scalar("First Epoch Training Loss History", loss.item(), batch_idx)
            writer.add_scalar("First Epoch Training Accuarcy History", acc/args.batch_size, batch_idx)
    
    return train_loss / len(train_loader), train_acc / len(train_loader.dataset)
        

def test(args, model, test_loader, criterion, epoch):
    model.eval()
    test_loss, test_acc = 0, 0
    pred_set = []
    label_set = []
    with torch.no_grad():
        tqdm_iter = tqdm(test_loader, desc="Epoch: {}/{} ({}%) |Testing loss: NaN".format(epoch, args.epoch, int(epoch/args.epoch)), leave=False)
        for batch_idx, (data, label, loc, agent_id) in enumerate(tqdm_iter):
            data, target = data.to(args.device), label.to(args.device)
            loc = loc.to(torch.float32).to(args.device)
            agent_id = agent_id.to(args.device)
            output = model(data, loc, agent_id)
            # loss = criterion(output, target)
            # acc = (output.argmax(dim=1) == target.argmax(dim=1)).float().sum().item()
            output = torch.sigmoid(output)
            num_pos = max(1.0, (target > 0).sum().item())
            loss = criterion(output, target, num_pos, args.alpha, args.gamma)
            acc = multi_label_acc(output=output, target=target, pos_thresh=args.acc_pos_thresh)
            
            test_loss += loss.cpu().item()
            test_acc += acc

            pred_set.append(output.cpu())
            label_set.append(target.cpu())

            tqdm_iter.set_description("Epoch: {}/{} ({}%) |Testing loss: {:.6f} |Testing Acc: {:.6f}".format(
                epoch, args.epoch, int(epoch/args.epoch), round(loss.item(), 6), round(acc / args.batch_size, 6)))
            
    return test_loss / len(test_loader), test_acc / len(test_loader.dataset), pred_set, label_set

