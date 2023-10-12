import os, sys
import random
import torch
import pickle
import numpy as np
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from core.data import CoresetSelection
from multiprocessing import Lock, Process, Queue, current_process, Manager
from core.data.sampling import GraphDensitySampler
import time
import queue
matplotlib.use('Agg')

def select_coreset(trainset, args):

    total_num = len(trainset)
    # total samples to be selected for coreset
    coreset_num = args.coreset_ratio * total_num

    # load data scores from training 100% data
    with open(args.data_score_path, 'rb') as f:
        data_score = pickle.load(f)

    # set descending=True if higher data score implies higher importance
    if args.coreset_key in ['entropy', 'forgetting', 'el2n', 'ssl']:
        print("Using descending order")
        args.data_score_descending = 1

    # get unique labels
    if args.label_balanced:
        if args.dataset in ['cifar10', 'cifar100', 'tinyimagenet', 'imagenet']:
            uniq_labels = set(trainset.targets)
            targets = trainset.targets

        else:
            uniq_labels = set(trainset.labels.tolist())
            targets = trainset.labels.tolist()

        uniq_labels = list(uniq_labels)
        if args.dataset in ['imagenet']:
            target_counts = Counter(targets)
            coreset_sizes_per_label = []
            for label in uniq_labels:
                coreset_sizes_per_label.append(int(coreset_num*target_counts[label]/len(targets)))
        else:
            coreset_sizes_per_label = int(coreset_num / len(uniq_labels))
            coreset_sizes_per_label = [coreset_sizes_per_label] * len(uniq_labels)
    else:
        uniq_labels = None
        targets = None

    score_index = None

    if args.coreset_mode == 'random':  # random sampling, baseline
        if args.label_balanced:
            coreset_num = args.coreset_ratio * total_num
            coreset_size_per_label = int(coreset_num / len(uniq_labels))
            coreset_index = []
            for label in uniq_labels:
                print("***************** Getting coresets for label", label, " ********************")
                # get sample for the label
                sample_idxs_by_label = torch.tensor(np.array([i for i in range(total_num) if targets[i] == label]))
                coreset_index_by_label = CoresetSelection.random_selection(total_num=len(sample_idxs_by_label),
                                                                           num=coreset_size_per_label)
                coreset_index.append(sample_idxs_by_label[coreset_index_by_label])
            coreset_index = torch.cat(coreset_index)
        else:
            coreset_index = CoresetSelection.random_selection(total_num=len(trainset),
                                                              num=args.coreset_ratio * len(trainset))

    if args.coreset_mode == 'coreset':  # for baseline methods other than CCS
        coreset_index = CoresetSelection.score_monotonic_selection(data_score=data_score, key=args.coreset_key,
                                                                   ratio=args.coreset_ratio,
                                                                   descending=(args.data_score_descending == 1),
                                                                   class_balanced=args.label_balanced)
        coreset_index_no_mis = coreset_index.numpy()

    if args.coreset_mode == 'moderate':  # for Moderate Coreset (Xia et al. 2023) baseline method
        assert args.feature_path
        features = np.load(args.feature_path)
        coreset_index = CoresetSelection.moderate_selection(data_score=data_score, ratio=args.coreset_ratio,
                                                            features=features)

    if args.coreset_mode == 'stratified':  # CCS Sampling, CCS + Graph (D2 Pruning)

        # remove low importance samples from data_score[args.coreset_key]
        mis_num = int(args.mis_ratio * total_num)
        data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=args.mis_key,
                                                                 mis_num=mis_num, mis_descending=args.mis_key in ['entropy', 'forgetting', 'el2n', 'ssl'],
                                                                 coreset_key=args.coreset_key)

        # features are needed if sampling is to be done based on graphs
        # if args.sampling_mode != 'random' or args.budget_mode == 'aucpr':
        if (args.sampling_mode == 'graph' and not args.precomputed_dists and not args.precomputed_neighbors) or args.aucpr or args.graph_score:
            assert args.feature_path
            features = np.load(args.feature_path)[score_index]
        else:
            features = None

        # remove mis_num samples beforehand if a data score in addition to  args.coreset_key is being used
        if args.budget_mode == 'confidence':
            data_score['confidence'] = data_score['confidence'][score_index]

        if args.label_balanced:

            uniq_labels = list(uniq_labels)
            def get_coreset_for_label(labels_to_accomplish, coreset_index):
                while True:
                    try:
                        n_label = labels_to_accomplish.get_nowait()
                        label = uniq_labels[n_label]
                        coreset_size_per_label = coreset_sizes_per_label[n_label]
                        print("***************** Getting coresets for label", label, " ********************")
                        # get sample for the label
                        sample_idxs_by_label = np.array([i for i, idx in enumerate(score_index) if targets[idx] == label])
                        data_score_by_label = {args.coreset_key: data_score[args.coreset_key][sample_idxs_by_label]}
                        if args.budget_mode == 'confidence':
                            data_score_by_label['confidence'] = data_score['confidence'][sample_idxs_by_label]
                        coreset_index_by_label, (pools, budgets) = CoresetSelection.stratified_sampling(data_score_by_label,
                                                                                                        coreset_size_per_label,
                                                                                                        args,
                                                                                                        data_embeds=None if features is None else
                                                                                                        features[
                                                                                                            sample_idxs_by_label])
                        assert len(coreset_index_by_label) <= coreset_size_per_label
                        print("Selected %s samples" % len(coreset_index_by_label))
                        assert all([targets[idx] == label for idx in score_index[sample_idxs_by_label[coreset_index_by_label]]])
                        coreset_index.extend(sample_idxs_by_label[coreset_index_by_label].tolist())
                        # coreset_index = np.concatenate((coreset_index, sample_idxs_by_label[coreset_index_by_label]), axis=0)
                    except queue.Empty:
                        break
                return True

            number_of_task = len(uniq_labels)
            number_of_processes = 8
            labels_to_accomplish = Queue()
            processes = []

            manager = Manager()
            coreset_index = manager.list()

            for i in range(number_of_task):
                labels_to_accomplish.put(i)

            # creating processes
            for w in range(number_of_processes):
                p = Process(target=get_coreset_for_label, args=(labels_to_accomplish, coreset_index))
                processes.append(p)
                p.start()

            # completing process
            for p in processes:
                p.join()
            coreset_index = np.array(coreset_index)

        else:
            coreset_index, _ = CoresetSelection.stratified_sampling(data_score, coreset_num, args,
                                                                    data_embeds=features)

        coreset_index_no_mis = np.array(coreset_index.copy())
        coreset_index = score_index[coreset_index]

    if args.coreset_mode == 'density':
        mis_num = int(args.mis_ratio * total_num)
        data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=args.mis_key, mis_num=mis_num,
                                                                 mis_descending=args.mis_key in ['entropy', 'forgetting', 'el2n', 'ssl'],
                                                                 coreset_key=args.coreset_key)

        # bins = np.load(args.bin_path)[score_index]
        bins = np.load(args.bin_path)
        assert len(bins) == len(score_index)

        if (args.sampling_mode == 'graph' and not args.precomputed_dists and not args.precomputed_neighbors) or args.aucpr:
            assert args.feature_path
            features = np.load(args.feature_path)[score_index]
        else:
            features = None
        if args.budget_mode == 'confidence':
            data_score['confidence'] = data_score['confidence'][score_index]

        if args.label_balanced:
            if args.dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
                uniq_labels = set(trainset.targets)
                targets = trainset.targets
            else:
                uniq_labels = set(trainset.labels.tolist())
                targets = trainset.labels.tolist()
            coreset_size_per_label = int(coreset_num / len(uniq_labels))
            coreset_index = []
            for n_label, label in enumerate(uniq_labels):
                coreset_size_per_label = coreset_sizes_per_label[n_label]
                print("***************** Getting coresets for label", label, " ********************")
                # get sample for the label
                sample_idxs_by_label = np.array([i for i, idx in enumerate(score_index) if targets[idx] == label])
                data_score_by_label = {args.coreset_key: data_score[args.coreset_key][sample_idxs_by_label]}
                if args.budget_mode == 'confidence':
                    data_score_by_label['confidence'] = data_score['confidence'][sample_idxs_by_label]
                coreset_index_by_label, _ = CoresetSelection.density_sampling(data_score_by_label,
                                                                              bins[sample_idxs_by_label],
                                                                              coreset_size_per_label, args,
                                                                              data_embeds=None if features is None else
                                                                              features[sample_idxs_by_label])
                assert all([targets[idx] == label for idx in score_index[sample_idxs_by_label[coreset_index_by_label]]])
                coreset_index = np.concatenate((coreset_index, sample_idxs_by_label[coreset_index_by_label]), axis=0)
        else:
            coreset_index, _ = CoresetSelection.density_sampling(data_score, bins, coreset_num, args,
                                                                 data_embeds=features)

        coreset_index_no_mis = coreset_index.copy()
        coreset_index = score_index[coreset_index]

    if args.coreset_mode == 'class':
        mis_num = int(args.mis_ratio * total_num)
        data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=args.mis_key,
                                                                 mis_num=mis_num,
                                                                 mis_descending=args.mis_key in ['entropy', 'forgetting', 'el2n', 'ssl'],
                                                                 coreset_key=args.coreset_key)
        coreset_num = int(args.coreset_ratio * total_num)
        if args.aucpr or (args.sampling_mode == 'graph' and not args.precomputed_dists and not args.precomputed_neighbors):
            assert args.feature_path
            print("Reading feature file")
            start_time = time.time()
            features = np.load(args.feature_path, mmap_mode='r')
            if mis_num:
                features = features[score_index]
            print("Finished reading feature file in ", time.time()-start_time, 'seconds')
        else:
            features = None
        if args.budget_mode == 'confidence':
            data_score['confidence'] = data_score['confidence'][score_index]
        targets = np.array(targets)[score_index]
        coreset_index, _ = CoresetSelection.density_sampling(data_score, targets, coreset_num, args, data_embeds=features)
        coreset_index_no_mis = np.array(coreset_index.copy())
        coreset_index = score_index[coreset_index]

    if args.coreset_mode == 'graph':
        mis_num = int(args.mis_ratio * total_num)
        _, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=args.mis_key,
                                                                 mis_num=mis_num,
                                                                 mis_descending=args.mis_key in ['entropy',
                                                                                                 'forgetting', 'el2n',
                                                                                                 'ssl'],
                                                                 coreset_key=args.coreset_key)

        np.save('./temp/imagenet/score_index_aum_%s.npy' % args.coreset_ratio, score_index)

        if (args.sampling_mode == 'graph' and not args.precomputed_dists and not args.precomputed_neighbors):
            assert args.feature_path
            features = np.load(args.feature_path)[score_index]
        else:
            features = None
        coreset_num = int(args.coreset_ratio * total_num)
        # load data scores from training 100% data
        sampling_method = GraphDensitySampler(X=features, y=None,
                                              gamma=args.gamma,
                                              seed=0, importance_scores=data_score[args.coreset_key], args=args)
                                              # n_neighbor=args.n_neighbor, graph_mode=args.graph_mode,
                                              # graph_sampling_mode=args.graph_sampling_mode,
                                              # precomputed_dists=args.precomputed_dists,
                                              # precomputed_neighbors=args.precomputed_neighbors
                                              # )
        coreset_index = sampling_method.select_batch_(coreset_num)
        coreset_index_no_mis = np.array(coreset_index.copy())
        coreset_index = score_index[coreset_index]
        graph_scores = sampling_method.starting_density

    if len(coreset_index) < coreset_num:
        if score_index is not None:
            extra_sample_set = list(set(score_index.tolist()).difference(set(coreset_index.tolist())))
            coreset_index = np.hstack((coreset_index, np.array(random.sample(extra_sample_set,
                                                                             k=int(min(len(extra_sample_set),
                                                                                   coreset_num - len(coreset_index)))))))
            print("Added extra %s samples" %  int(min(len(extra_sample_set), coreset_num-len(coreset_index))))
            print(coreset_index.shape)

    # coreset_index = np.array(coreset_index.detach().cpu())
    trainset = torch.utils.data.Subset(trainset, coreset_index)
    print("Pruned %s samples in original train set to %s" % (total_num, len(trainset)))

    out_file = '../data-pruning-analysis/%s_%s.png' % (args.dataset, args.task_name)

    if args.coreset_mode == 'graph' and args.coreset_key == 'unity':
        entropy = plot_score_distribution(graph_scores, coreset_index_no_mis.astype(np.int32),
                                          out_file,
                                          args.stratas, args.coreset_key, args)
    else:
        # load data scores from training 100% data
        with open(args.data_score_path, 'rb') as f:
            data_score = pickle.load(f)
        entropy = plot_score_distribution(data_score[args.coreset_key].numpy(), coreset_index.numpy().astype(np.int32), out_file,
                            args.stratas, args.coreset_key, args)
    return trainset, np.array(coreset_index), entropy


def plot_score_distribution(scores, coreset_index, out_file, n_bins=25, coreset_key='forgetting', args=None):

    total_num = len(scores)
    coreset_num = len(coreset_index)

    score_min = np.min(scores)
    score_max = np.max(scores)

    if coreset_key == 'accumulated_margin':
        scores = -(scores - score_max)
        score_min = np.min(scores)
        score_max = np.max(scores)

    bins = np.linspace(score_min, score_max, n_bins + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3,4))
    counts = [np.sum((scores >= bins[i-1]) & (scores < bins[i]))/total_num for i in range(1, n_bins+1)]
    coreset_scores = scores[coreset_index]
    coreset_counts = [np.sum((coreset_scores >= bins[i-1]) & (coreset_scores < bins[i]))/coreset_num for i in range(1, n_bins+1)]
    coreset_counts = [max(1e-6, c) for c in coreset_counts]
    entropy = -(np.array(coreset_counts) * np.log(np.abs(np.array(coreset_counts)))).sum()
    print('Rendering with min score %s and max score %s' % (score_min, score_max))
    print(bins)
    print(coreset_counts)
    if coreset_key in ['forgetting']:
        ax1.bar(bins[1:], counts)
        ax2.bar(bins[1:], coreset_counts)
    else:
        ax1.bar(bins[1:], counts)
        ax2.bar(bins[1:], coreset_counts)
        # ax1.plot(bins[1:], counts)
        # ax2.plot(bins[1:], coreset_counts)
    ax1.set_ylim(0, max(counts) + 0.1)
    ax2.set_ylim(0, max(counts) + 0.1)
    ax2.set_xlabel('Difficulty Score', fontsize=10)
    ax1.set_ylabel('Fraction of Dataset', fontsize=10)
    ax2.set_ylabel('Fraction of Coreset', fontsize=10)
    ax1.set_title('Full dataset distribution', fontsize=10)
    ax2.set_title(r'$k$=%s, $\gamma_{r}$=%s, Entropy=%s' % (args.n_neighbor, round(args.gamma, 1), round(entropy, 3)), fontsize=10)
    #ax2.set_title(r'$k$=%s, $\gamma_{r}$=%s, Entropy=%s' % (100, 0.1, round(entropy, 3)), fontsize=10)
    ax1.grid(True, linestyle='--')
    ax2.grid(True, linestyle='--')
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    return entropy


def visualize_coreset(pools, budgets, score_index):

    ccs_idxs = []
    for i, p in enumerate(pools):
        if budgets[i] == 0:
            continue
        else:
            pool = score_index[sample_idxs_by_label[p]].tolist()
            if type(pool) == int:
                pool = [pool]
            idxs = random.sample(list(range(len(pool))), k=budgets[i])
            idxs.sort()
            ccs_idxs.extend([pool[idx] for idx in idxs])
    selected_idxs = score_index[sample_idxs_by_label[coreset_index_by_label]]
    # ccs_idxs = random.sample(list(range(len(pools))), k=50)
    # ccs_idxs.sort()
    # ccs_idxs = [all_pools[idx] for idx in ccs_idxs]

    fig = plt.figure(figsize=(10, 5))
    for i, idx in enumerate(selected_idxs):
        # Adds a subplot at the 1st position
        fig.add_subplot(5, 10, i+1)
        plt.imshow(trainset[idx][0])
        plt.axis('off')
    plt.savefig('../out/d2_%s.png' % n_label)
    fig = plt.figure(figsize=(10, 5))
    for i, idx in enumerate(ccs_idxs):
        # Adds a subplot at the 1st position
        fig.add_subplot(5, 10, i+1)
        plt.imshow(trainset[idx][0])
        plt.axis('off')
    plt.savefig('../out/ccs_%s.png' % n_label)
