import torch
import os

def prediction_correct(true, preds):
    """
    Computes prediction_hit.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Prediction_hit for each img.
    """
    rst = (torch.softmax(preds, dim=1).argmax(dim=1) == true)
    return rst.detach().cpu().type(torch.int)

def get_model_directory(base_dir, model_name):
    model_dir = os.join(base_dir, model_name)
    ckpt_dir = os.join(model_dir, 'ckpt')
    data_dir = os.join(model_dir, 'data')
    log_dir = os.join(model_dir, 'log')

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return ckpt_dir, data_dir, log_dir

def l2_distance(tensor1, tensor2):
    dist = (tensor1 - tensor2).pow(2).sum().sqrt()
    return dist

def accuracy(output, target, topk=1):
    """Computes the precision@k for the specified values of k"""
    maxk = topk
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct.float().sum()
    acc = correct_k * (100.0 / batch_size)

    return acc, correct_k.item()