import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle

from core.model_generator import wideresnet, preact_resnet, resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset, TinyImageNetDataset
from core.utils import print_training_info, StdRedirect

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

######################### Data Setting #########################
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny', 'svhn', 'cinic10', 'tinyimagenet'])

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

######################### Other Options ###########################
parser.add_argument('--feature', action="store_true",
                    help="Set true if feature maps need to be extracted")

args = parser.parse_args()

######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
data_score_path = os.path.join(task_dir, f'data-score-{args.task_name}.pickle')
train_feature_path = os.path.join(task_dir, f'train-features-conv3-{args.task_name}.npy')
test_feature_path = os.path.join(task_dir, f'test-features-{args.task_name}.npy')


######################### Print setting #########################
print_training_info(args, all=True)

#########################
dataset = args.dataset
if dataset in ['cifar10', 'svhn', 'cinic10']:
    num_classes=10
elif dataset == 'tinyimagenet':
    num_classes = 200
elif dataset == 'cifar100':
    num_classes=100


######################### Ftn definition #########################
"""Calculate loss and entropy"""
def post_training_metrics(model, dataloader, data_importance, device):
    model.eval()
    data_importance['entropy'] = torch.zeros(len(dataloader.dataset))
    data_importance['loss'] = torch.zeros(len(dataloader.dataset))
    data_importance['confidence'] = torch.zeros(len(dataloader.dataset))

    for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        prob = nn.Softmax(dim=1)(logits)

        entropy = -1 * prob * torch.log(prob + 1e-10)
        entropy = torch.sum(entropy, dim=1).detach().cpu()

        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets).detach().cpu()
        confidence = prob[torch.arange(0, logits.shape[0]).to(device), targets].detach().cpu()
        # print(targets.shape, prob.shape, confidence.shape)

        data_importance['entropy'][idx] = entropy
        data_importance['loss'][idx] = loss
        data_importance['confidence'][idx] = confidence

"""Calculate td metrics"""
def training_dynamics_metrics(td_log, dataset, data_importance):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        correctness = (predicted == label).type(torch.int)
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        record_training_dynamics(item)

"""Calculate td metrics"""
def EL2N(td_log, dataset, data_importance, max_epoch=10):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = td_log['idx'].type(torch.long)

        label = targets[index]

        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
        el2n_score = torch.sqrt(l2_loss(label_onehot,output).sum(dim=1))

        data_importance['el2n'][index] += el2n_score

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)
        if item['epoch'] == max_epoch:
            return
        record_training_dynamics(item)

"""Extract feature map"""
def feature_maps(model, dataloader, device, layer=4):
    model.eval()
    features = []
    for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        feats = model.feature_map(inputs, layer=layer)
        features.append(feats.detach().cpu().numpy())

    features = np.concatenate(features)
    return features

#########################

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_identical = transforms.Compose([
            transforms.ToTensor(),
        ])

data_dir =  os.path.join(args.data_dir, dataset)
print(f'dataset: {dataset}')
if dataset == 'cifar10':
    trainset = CIFARDataset.get_cifar10_train(data_dir, transform = transform_identical)
    testset = CIFARDataset.get_cifar10_test(data_dir)
elif dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(data_dir, transform = transform_identical)
    testset = CIFARDataset.get_cifar100_test(data_dir)
elif dataset == 'svhn':
    trainset = SVHNDataset.get_svhn_train(data_dir, transform = transform_identical)
    testset = SVHNDataset.get_svhn_test(data_dir)
elif args.dataset == 'cinic10':
    trainset = CINIC10Dataset.get_cinic10_train(data_dir, transform = transform_identical)
    testset = CINIC10Dataset.get_cinic10_test(data_dir)
elif args.dataset == 'tinyimagenet':
    trainset = TinyImageNetDataset.get_tinyimagenet_train(data_dir, transform= transform_identical)
    testset = TinyImageNetDataset.get_tinyimagenet_test(data_dir)
else:
    raise ValueError

trainset = IndexDataset(trainset)
print(len(trainset))

data_importance = {}

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, num_workers=16)

model = resnet('resnet18', num_classes=num_classes, device=device)
model = model.to(device)

print(f'Ckpt path: {ckpt_path}.')
checkpoint = torch.load(ckpt_path)['model_state_dict']
model.load_state_dict(checkpoint)
model.eval()

with open(td_path, 'rb') as f:
     pickled_data = pickle.load(f)

training_dynamics = pickled_data['training_dynamics']

post_training_metrics(model, trainloader, data_importance, device)
training_dynamics_metrics(training_dynamics, trainset, data_importance)
EL2N(training_dynamics, trainset, data_importance, max_epoch=10)

print(f'Saving data score at {data_score_path}')
with open(data_score_path, 'wb') as handle:
    pickle.dump(data_importance, handle)

if args.feature:
    print(f'Saving train features at {train_feature_path}')
    features = feature_maps(model, trainloader, device)
    print(features.shape)
    np.save(train_feature_path, features)
    print(f'Saving test features at {test_feature_path}')
    testset = IndexDataset(testset)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    features = feature_maps(model, testloader, device)
    np.save(test_feature_path, features)
