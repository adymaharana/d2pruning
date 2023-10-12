import time
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, random
import argparse
from datetime import datetime
from selection_mp import select_coreset

from core.model_generator import wideresnet, preact_resnet, resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset, TinyImageNetDataset
from core.utils import print_training_info, StdRedirect
import numpy as np

model_names = ['resnet18', 'wrn-34-10', 'preact_resnet18']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')

######################### Training Setting #########################
parser.add_argument('--epochs', type=int, metavar='N',
                    help='The number of epochs to train a model.')
parser.add_argument('--iterations', type=int, metavar='N',
                    help='The number of iteration to train a model; conflict with --epoch.')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--network', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'cinic10', 'tinyimagenet'])

######################### Print Setting #########################
parser.add_argument('--iterations-per-testing', type=int, default=800, metavar='N',
                    help='The number of iterations for testing model')

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')

######################### Coreset Setting #########################
parser.add_argument('--coreset', action='store_true', default=False)
parser.add_argument('--coreset-only', action='store_true', default=False)
parser.add_argument('--coreset-mode', type=str, choices=['random', 'coreset', 'stratified', 'density', 'class', 'graph'])
parser.add_argument('--sampling-mode', type=str, choices=['kcenter', 'random', 'graph'])
parser.add_argument('--budget-mode', type=str, choices=['uniform', 'density', 'confidence', 'aucpr'])

parser.add_argument('--data-score-path', type=str)
parser.add_argument('--bin-path', type=str)
parser.add_argument('--feature-path', type=str)
parser.add_argument('--coreset-key', type=str)
parser.add_argument('--data-score-descending', type=int, default=0,
                    help='Set 1 to use larger score data first.')
parser.add_argument('--class-balanced', type=int, default=0,
                    help='Set 1 to use the same class ratio as to the whole dataset.')
parser.add_argument('--coreset-ratio', type=float)
parser.add_argument('--label-balanced', action='store_true', default=False)

######################## CCS Setting ###########################################
parser.add_argument('--aucpr', action='store_true', default=False)
parser.add_argument('--stratas', type=int, default=50)
parser.add_argument('--graph-score', action='store_true', default=False)

######################## Graph Sampling Setting ################################
parser.add_argument('--n-neighbor', type=int, default=10)
parser.add_argument('--gamma', type=float, default=-1)
parser.add_argument('--graph-mode', type=str, default='')
parser.add_argument('--graph-sampling-mode', type=str, default='')
parser.add_argument('--precomputed-dists', type=str, default='')
parser.add_argument('--precomputed-neighbors', type=str, default='')


#### Double-end Pruning Setting ####
parser.add_argument('--mis-key', type=str, default='accumulated_margin')
parser.add_argument('--mis-data-score-descending', type=int, default=0,
                    help='Set 1 to use larger score data first.')
parser.add_argument('--mis-ratio', type=float)

#### Reversed Sampling Setting ####
parser.add_argument('--reversed-ratio', type=float,
                    help="Ratio for the coreset, not the whole dataset.")

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

######################### Setting for Future Use #########################
# parser.add_argument('--ckpt-name', type=str, default='model.ckpt',
#                     help='The name of the checkpoint.')
# parser.add_argument('--lr-scheduler', choices=['step', 'cosine'])
# parser.add_argument('--network', choices=model_names, default='resnet18')
# parser.add_argument('--pretrained', action='store_true')
# parser.add_argument('--augment', choices=['cifar10', 'rand'], default='cifar10')

args = parser.parse_args()
start_time = datetime.now()

assert args.epochs is None or args.iterations is None, "Both epochs and iterations are used!"


print(f'Dataset: {args.dataset}')
######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
os.makedirs(task_dir, exist_ok=True)
last_ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
best_ckpt_path = os.path.join(task_dir, f'ckpt-best.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
log_path = os.path.join(task_dir, f'log-train-{args.task_name}.log')
coreset_index_path = os.path.join(task_dir, f'coreset-{args.task_name}.npy')

######################### Print setting #########################
sys.stdout=StdRedirect(log_path)
print_training_info(args, all=True)
#########################
print(f'Last ckpt path: {last_ckpt_path}')

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.path.join(args.data_dir, args.dataset)
print(f'Data dir: {data_dir}')

if args.dataset == 'cifar10':
    trainset = CIFARDataset.get_cifar10_train(data_dir)
elif args.dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(data_dir)
    # trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
elif args.dataset == 'svhn':
    trainset = SVHNDataset.get_svhn_train(data_dir)
elif args.dataset == 'cinic10':
    trainset = CINIC10Dataset.get_cinic10_train(data_dir)
elif args.dataset == 'tinyimagenet':
    trainset = TinyImageNetDataset.get_tinyimagenet_train(data_dir)
else:
    raise ValueError

######################### Coreset Selection #########################
coreset_key = args.coreset_key
coreset_ratio = args.coreset_ratio
coreset_descending = (args.data_score_descending == 1)
total_num = len(trainset)

if args.coreset:
    start_time = time.time()
    trainset, coreset_index, _ = select_coreset(trainset, args)
    print("Completed coreset selection in %s seconds" % (time.time()-start_time))
    np.save(coreset_index_path, np.array(coreset_index))
    if args.coreset_only:
        sys.exit()
######################### Coreset Selection end #########################

trainset = IndexDataset(trainset)
print(len(trainset))

if args.dataset == 'cifar10':
    testset = CIFARDataset.get_cifar10_test(data_dir)
elif args.dataset == 'cifar100':
    testset = CIFARDataset.get_cifar100_test(data_dir)
elif args.dataset == 'svhn':
    testset = SVHNDataset.get_svhn_test(data_dir)
elif args.dataset == 'cinic10':
    testset = CINIC10Dataset.get_cinic10_test(data_dir)
elif args.dataset == 'tinyimagenet':
    testset = TinyImageNetDataset.get_tinyimagenet_test(data_dir)
else:
    raise ValueError

print("%s samples in test set" % len(testset))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=True, num_workers=16)

iterations_per_epoch = len(trainloader)
if args.iterations is None:
    num_of_iterations = iterations_per_epoch * args.epochs
else:
    num_of_iterations = args.iterations

if args.dataset in ['cifar10', 'svhn', 'cinic10']:
    num_classes=10
elif args.dataset == 'tinyimagenet':
    num_classes = 200
else:
    num_classes=100

if args.network == 'resnet18':
    print('resnet18')
    model = resnet('resnet18', num_classes=num_classes, device=device)
if args.network == 'resnet50':
    print('resnet50')
    model = resnet('resnet50', num_classes=num_classes, device=device)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_iterations, eta_min=1e-4)

epoch_per_testing = args.iterations_per_testing // iterations_per_epoch

print(f'Total epoch: {num_of_iterations // iterations_per_epoch}')
print(f'Iterations per epoch: {iterations_per_epoch}')
print(f'Total iterations: {num_of_iterations}')
print(f'Epochs per testing: {epoch_per_testing}')

trainer = Trainer()
TD_logger = TrainingDynamicsLogger()

best_acc = 0
best_epoch = -1

current_epoch = 0
while num_of_iterations > 0:
    iterations_epoch = min(num_of_iterations, iterations_per_epoch)
    trainer.train(current_epoch, -1, model, trainloader, optimizer, criterion, scheduler, device, TD_logger=TD_logger, log_interval=60, printlog=True)

    num_of_iterations -= iterations_per_epoch

    if current_epoch % epoch_per_testing == 0 or num_of_iterations == 0:
        test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)

    current_epoch += 1
    # scheduler.step()

# last ckpt testing
test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True)
if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)
print('==========================')
print(f'Best acc: {best_acc * 100:.2f}')
print(f'Best acc: {best_acc}')
print(f'Best epoch: {best_epoch}')
print(best_acc)
######################### Save #########################
state = {
    'model_state_dict': model.state_dict(),
    'epoch': current_epoch - 1
}
torch.save(state, last_ckpt_path)
TD_logger.save_training_dynamics(td_path, data_name=args.dataset)

print('Total time consumed:', (time.time() - start_time))