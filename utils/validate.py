import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging
from tqdm import tqdm  # [新增] 引入进度条库


# validate the algorithm by AUC, accuracy and f1 score on val/test datasets

def algorithm_validate(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        # [新增] 使用 tqdm 包裹 data_loader，显示进度条
        # desc: 显示前缀，例如 "Validating (test)"
        # leave=False: 跑完后清除进度条，保持终端整洁
        # dynamic_ncols=True: 自动适应终端宽度
        loader_bar = tqdm(data_loader, desc=f"Validating ({val_type})", leave=False, dynamic_ncols=True)

        for image, label, domain, _ in loader_bar:
            image = image.cuda()
            label = label.cuda().long()

            output = algorithm.predict(image)
            batch_loss = criterion(output, label).item()
            loss += batch_loss

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())

            # [可选] 在进度条尾部显示当前的 Loss，让你知道模型还在“动”
            loader_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

        # 循环结束，计算指标
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')

        # 处理可能缺类导致的 AUC 计算报错
        try:
            auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')
        except ValueError:
            auc_ovo = 0.0

        loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            # 记录到 Tensorboard
            if writer is not None:
                writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
                writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
                writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
                writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)

                # 记录到日志文件
            logging.info('{} - epoch: {}, loss: {:.4f}, acc: {:.4f}, auc: {:.4f}, F1: {:.4f}.'.format
                         (val_type, epoch, loss, acc, auc_ovo, f1))

    algorithm.train()
    return auc_ovo, loss