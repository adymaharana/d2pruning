import random, math
import torch
import numpy as np
from collections import Counter
from .sampling import kCenterGreedy, GraphDensitySampler
from .aucpr import get_aucpr
from tqdm import tqdm
from multiprocessing import Lock, Process, Queue, current_process, Manager
import queue

def get_median(features, targets):
    # get the median feature vector of each class
    num_classes = len(np.unique(targets, axis=0))
    prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)

    for i in range(num_classes):
        prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
    return prot


def get_distance(features, labels):

    prots = get_median(features, labels)
    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))

    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        prots_for_each_example[(labels == i).nonzero()[0], :] = prots[i]
    distance = np.linalg.norm(features - prots_for_each_example, axis=1)

    return distance


def bin_allocate(num, bins, mode='uniform', initial_budget=None):
    sorted_index = torch.argsort(bins)
    sort_bins = bins[sorted_index]

    num_bin = bins.shape[0]

    rest_exp_num = num
    budgets = []
    for i in range(num_bin):
        if sort_bins[i] == 0:
            budgets.append(0)
            continue
        # rest_bins = num_bin - i
        rest_bins = torch.count_nonzero(sort_bins[i:])
        if mode == 'uniform':
            avg = rest_exp_num // rest_bins
            cur_num = min(sort_bins[i].item(), avg)
            rest_exp_num -= cur_num
        else:
            avg = initial_budget[sorted_index[i]]
            cur_num = min(sort_bins[i].item(), avg)
            delta = int((avg - cur_num)/max(1, (rest_bins - 1)))
            # print("At index %s, changing budget from %s to %s and reallocating %s to %s bins" % (i, avg, cur_num, delta, rest_bins-1))
            for j in range(i+1, num_bin):
                initial_budget[sorted_index[j]] += delta
        budgets.append(cur_num)

    budgets = torch.tensor(budgets)
    if torch.sum(budgets) < num: # TODO: check again
        delta = num - torch.sum(budgets)
        i = 1
        while delta and i <= num_bin:
            if budgets[-i] < sort_bins[-i]:
                budgets[-i] += 1
                delta -= 1
            i += 1

    rst = torch.zeros((num_bin,)).type(torch.int)
    rst[sorted_index] = torch.tensor(budgets).type(torch.int)

    assert all([b<= r for r, b in zip(bins, rst)]), ([(r.item(),b.item()) for r, b in zip(bins, rst)], bins, [x.item() for x in torch.tensor(budgets)[sorted_index]])
    return rst


class CoresetSelection(object):

    @staticmethod
    def moderate_selection(data_score, ratio, features):

        def get_prune_idx(rate, distance):
            rate = 1-rate
            low = 0.5 - rate / 2
            high = 0.5 + rate / 2

            sorted_idx = distance.argsort()
            low_idx = round(distance.shape[0] * low)
            high_idx = round(distance.shape[0] * high)

            ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))

            return ids

        targets_list = data_score['targets']
        distance = get_distance(features, targets_list)
        ids = get_prune_idx(ratio, distance)

        return ids


    @staticmethod
    def score_monotonic_selection(data_score, key, ratio, descending, class_balanced):
        score = data_score[key]
        score_sorted_index = score.argsort(descending=descending)
        total_num = ratio * data_score[key].shape[0]
        print("Selecting from %s samples" % total_num)
        if class_balanced:
            print('Class balance mode.')
            all_index = torch.arange(data_score['targets'].shape[0])
            #Permutation
            selected_index = []
            targets_list = data_score['targets'][score_sorted_index]
            targets_unique = torch.unique(targets_list)
            for target in targets_unique:
                target_index_mask = (targets_list == target)
                target_index = all_index[target_index_mask]
                targets_num = target_index_mask.sum()
                target_coreset_num = targets_num * ratio
                selected_index = selected_index + list(target_index[:int(target_coreset_num)])
                print("Selected %s samples for %s label" % (len(selected_index), target))
            selected_index = torch.tensor(selected_index)
            print(f'High priority {key}: {score[score_sorted_index[selected_index][:15]]}')
            print(f'Low priority {key}: {score[score_sorted_index[selected_index][-15:]]}')
            return score_sorted_index[selected_index]
        else:
            print(f'High priority {key}: {score[score_sorted_index[:15]]}')
            print(f'Low priority {key}: {score[score_sorted_index[-15:]]}')
            return score_sorted_index[:int(total_num)]

    @staticmethod
    def mislabel_mask(data_score, mis_key, mis_num, mis_descending, coreset_key):
        mis_score = data_score[mis_key]
        mis_score_sorted_index = mis_score.argsort(descending=mis_descending)
        hard_index = mis_score_sorted_index[:mis_num]
        print(f'Bad data -> High priority {mis_key}: {data_score[mis_key][hard_index][:15]}')
        print(f'Prune {hard_index.shape[0]} samples.')

        easy_index = mis_score_sorted_index[mis_num:]
        data_score[coreset_key] = data_score[coreset_key][easy_index]

        return data_score, easy_index

    @staticmethod
    # def stratified_sampling(data_score, coreset_key, coreset_num, budget='uniform',
    #                         sampling='random', data_embeds=None,
    #                         n_neighbor=10, median=False, stratas=50):
    def stratified_sampling(data_score, coreset_num, args, data_embeds=None):

        if args.sampling_mode == 'graph' and args.coreset_key in ['accumulated_margin']: # TODO: check again
            score = data_score[args.coreset_key]
            min_score = torch.min(score)
            max_score = torch.max(score)
            score = score - min_score
            data_score[args.coreset_key] = -score

        print('Using stratified sampling...')
        score = data_score[args.coreset_key]
        if args.graph_score:
            graph = GraphDensitySampler(X=data_embeds, y=None, gamma=args.gamma,
                                                  seed=0, importance_scores=score, args=args)
                                        #           n_neighbor=args.n_neighbor, graph_mode=args.graph_mode,
                                        # graph_sampling_mode=args.graph_sampling_mode,
                                        # precomputed_dists=args.precomputed_dists,
                                        # precomputed_neighbors=args.precomputed_neighbors)
            score = torch.tensor(graph.graph_density)

        total_num = len(score)
        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        print("Min score: %s, max score: %s" % (min_score.item(), max_score.item()))
        step = (max_score - min_score) / args.stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = []
        ##### calculate number of samples in each strata #####
        for i in range(args.stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)
        strata_num = torch.tensor(strata_num)

        if args.budget_mode == 'uniform':
            budgets = bin_allocate(coreset_num, strata_num)
        elif args.budget_mode == 'confidence':
            confs = data_score['confidence']
            mean_confs = []
            for i in range(args.stratas):
                start, end = bin_range(i)
                sample_idxs = torch.logical_and(score >= start, (score < end)).nonzero().squeeze()
                if sample_idxs.size()[0] != 0:
                    mean_confs.append(1-torch.mean(confs[sample_idxs]).item())
                else:
                    mean_confs.append(0)
            total_conf = np.sum(mean_confs)
            budgets = [int(n*coreset_num/total_conf) for n in mean_confs]
            print("Initial budget", budgets)
            budgets = bin_allocate(coreset_num, strata_num, mode='confidence', initial_budget=budgets)
        elif args.budget_mode == 'aucpr':
            budgets = bin_allocate(coreset_num, strata_num)
            sample_index = torch.arange(data_score[args.coreset_key].shape[0])
            aucpr_values = []
            min_budgets = {}
            for i in tqdm(range(args.stratas), desc='Getting k-centers for aucpr-based budgeting'):
                if budgets[i] == 0:
                    aucpr_values.append(0)
                    continue
                start, end = bin_range(i)
                mask = torch.logical_and(score >= start, score < end)
                pool = sample_index[mask]

                if args.sampling_mode == 'random':
                    rand_index = torch.randperm(pool.shape[0])
                    selected_idxs = [idx.item() for idx in rand_index[:budgets[i]]]
                elif args.sampling_mode == 'kcenter':
                    sampling_method = kCenterGreedy(X=data_embeds[pool], y=None, seed=0)
                    selected_idxs = sampling_method.select_batch_(None, budgets[i])

                elif args.sampling_mode == 'graph':
                    if pool.shape[0] <= args.n_neighbor:
                        rand_index = torch.randperm(pool.shape[0])
                        selected_idxs = rand_index[:budgets[i]].numpy().tolist()
                    else:
                        sampling_method = GraphDensitySampler(X=None if data_embeds is None else data_embeds[pool], y=None, gamma=args.gamma,
                                                              seed=0, importance_scores=score[pool], args=args)
                                                              # n_neighbor=args.n_neighbor, graph_mode=args.graph_mode,
                                                              # graph_sampling_mode=args.graph_sampling_mode,
                                                              # precomputed_dists=args.precomputed_dists,
                                                              # precomputed_neighbors=args.precomputed_neighbors
                                                              # )
                        selected_idxs = sampling_method.select_batch_(budgets[i])
                else:
                    raise ValueError

                kcenters = pool[selected_idxs]
                non_coreset = list(set(pool.tolist()).difference(set(kcenters.tolist())))
                aucpr = get_aucpr(data_embeds[kcenters], data_embeds[non_coreset])
                aucpr_values.append(round(aucpr, 3))
                if aucpr == 0:
                    min_budgets[i] = budgets[i]

            print("Initial AUCpr values: ", aucpr_values)
            print("Initial mean AUCpr: ", np.mean(aucpr_values))
            total_aucpr = np.sum(aucpr_values)
            print("Uniform budget", budgets)
            if total_aucpr == 0:
                pass
            else:
                budgets = [int(n*(coreset_num-sum(min_budgets.values()))/total_aucpr) if i not in min_budgets
                           else min_budgets[i] for i, n in enumerate(aucpr_values)]
                print("Initial budget", budgets)
                budgets = bin_allocate(coreset_num, strata_num, mode='aucpr', initial_budget=budgets)
        else:
            raise ValueError
        # assert budgets.sum().item() == coreset_num, (budgets.sum(), coreset_num)
        print(budgets, budgets.sum())

        ##### sampling in each strata #####
        selected_index = []
        sample_index = torch.arange(data_score[args.coreset_key].shape[0])

        pools, kcenters = [], []
        for i in tqdm(range(args.stratas), desc='sampling from each strata'):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]
            pools.append(pool)

            if len(pool.numpy().tolist()) == 0 or budgets[i] == 0:
                continue
            if args.sampling_mode == 'random':
                rand_index = torch.randperm(pool.shape[0])
                selected_idxs = [idx.item() for idx in rand_index[:budgets[i]]]
            elif args.sampling_mode == 'kcenter':
                sampling_method = kCenterGreedy(X=data_embeds[pool], y=None, seed=0)
                selected_idxs = sampling_method.select_batch_(None, budgets[i])
            elif args.sampling_mode == 'graph':
                if pool.shape[0] <= args.n_neighbor: # if num of samples are less than size of graph, select all
                    rand_index = torch.randperm(pool.shape[0])
                    selected_idxs = rand_index[:budgets[i]].numpy().tolist()
                else:
                    sampling_method = GraphDensitySampler(X=None if data_embeds is None else data_embeds[pool], y=None, gamma=args.gamma, seed=0,
                                                          importance_scores=score[pool], args=args)
                                                          # n_neighbor=args.n_neighbor, graph_mode=args.graph_mode,
                                                          # graph_sampling_mode=args.graph_sampling_mode,
                                                          # precomputed_dists=args.precomputed_dists,
                                                          # precomputed_neighbors=args.precomputed_neighbors
                                                          # )
                    selected_idxs = sampling_method.select_batch_(budgets[i])
            else:
                raise ValueError
            kcenters.append(pool[selected_idxs])

        if args.aucpr:
            final_aucpr_values = []
            for pool, samples in zip(pools, kcenters):
                if len(pool.numpy().tolist()) == 0 or budgets[i] == 0:
                    final_aucpr_values.append(0.0)
                non_coreset = list(set(pool.tolist()).difference(set(samples.tolist())))
                if len(non_coreset) == 0:
                    aucpr = 0
                else:
                    aucpr = get_aucpr(data_embeds[kcenters], data_embeds[non_coreset])
                final_aucpr_values.append(round(aucpr, 3))
            print("Final AUCpr values: ", final_aucpr_values)
            print("Final mean AUCpr: ", np.mean(final_aucpr_values))

        for samples in kcenters:
            selected_index += samples

        return selected_index, (pools, budgets)

    @staticmethod
    def density_sampling(data_score, bins, coreset_num, args, data_embeds=None):

        if args.sampling_mode == 'graph' and args.coreset_key in ['accumulated_margin']: # TODO: check again
            score = data_score[args.coreset_key]
            min_score = torch.min(score)
            max_score = torch.max(score)
            score = score - min_score
            data_score[args.coreset_key] = score

        hist = np.histogram(bins, bins=np.arange(0, np.amax(bins) + 2, 1))[0]
        n_bins = np.amax(bins) + 1
        bin_pop_density = Counter(hist.tolist())
        print("Frequency of bin counts", bin_pop_density.most_common(20))

        non_empty_bins = np.where(hist != 0)[0]
        print("Skipping %s empty bins in total %s bins" % ((n_bins - non_empty_bins.shape[0]), n_bins))

        strata_num = []
        bin2size = {bin_idx: hist[bin_idx] for bin_idx in non_empty_bins}
        ##### calculate number for each strata #####
        for i in non_empty_bins:
            strata_num.append(bin2size[i])
        strata_num = torch.tensor(strata_num)

        if args.budget_mode == 'density':
            total_num = sum(list(bin2size.values()))
            bin2budget = {bin_idx: math.ceil(bin2size[bin_idx]*coreset_num/total_num) for bin_idx in non_empty_bins}
        elif args.budget_mode == 'uniform':
            budgets = bin_allocate(coreset_num, strata_num)
            bin2budget = {bin_i: budgets[i] for i, bin_i in enumerate(non_empty_bins)}
        elif args.budget_mode == 'confidence':
            confs = data_score['confidence']
            mean_confs = []
            strata_num = []
            for i in non_empty_bins:
                sample_idxs = np.where(bins == i)[0]
                mean_confs.append(1 - torch.mean(confs[sample_idxs]).item())
                strata_num.append(bin2size[i])
            strata_num = torch.tensor(strata_num)
            total_conf = np.sum(mean_confs)
            budgets = [int(n * coreset_num / total_conf) for n in mean_confs]
            print("Initial budget", budgets)
            budgets = bin_allocate(coreset_num, strata_num, mode='confidence', initial_budget=budgets)
            print("Final budget", budgets)
            bin2budget = {bin_idx: budgets[i] for i, bin_idx in enumerate(non_empty_bins)}
        elif args.budget_mode == 'aucpr':
            budgets = bin_allocate(coreset_num, strata_num)
            aucpr_values = []
            min_budgets = {}
            for i, bin_idx in tqdm(enumerate(non_empty_bins), desc='Getting k-centers for aucpr-based budgeting'):
                if budgets[i] == 0:
                    aucpr_values.append(0)
                    continue
                sample_idxs = np.where(bins == bin_idx)[0]

                if args.sampling_mode == 'random':
                    rand_index = np.random.permutation(sample_idxs.shape[0])
                    selected_idx = rand_index[:budgets[i]]
                elif args.sampling_mode == 'kcenter':
                    sampling_method = kCenterGreedy(X=data_embeds[sample_idxs], y=None, seed=0)
                    selected_idx = sampling_method.select_batch_(None, budgets[i])
                elif args.sampling_mode == 'graph':
                    if sample_idxs.shape[0] <= 10:
                        selected_idx = np.random.permutation(sample_idxs.shape[0])
                    else:
                        sampling_method = GraphDensitySampler(X=None if data_embeds is None else data_embeds[sample_idxs], y=None, gamma=args.gamma, seed=0,
                                                              importance_scores=data_score['forgetting'][sample_idxs], args=args)
                                                              # n_neighbor=args.n_neighbor, graph_mode=args.graph_mode,
                                                              # graph_sampling_mode=args.graph_sampling_mode,
                                                              # precomputed_dists=args.precomputed_dists,
                                                              # precomputed_neighbors=args.precomputed_neighbors
                                                              # )
                        selected_idx = sampling_method.select_batch_(budgets[i])
                else:
                    raise ValueError

                kcenters = sample_idxs[selected_idx]
                non_coreset = list(set(sample_idxs.tolist()).difference(set(kcenters.tolist())))
                aucpr = get_aucpr(data_embeds[kcenters], data_embeds[non_coreset])
                aucpr_values.append(aucpr)
                if aucpr == 0:
                    min_budgets[bin_idx] = budgets[i]

            print("Initial AUCpr values: ", aucpr_values)
            print("Initial mean AUCpr: ", np.mean(aucpr_values))
            total_aucpr = np.sum(aucpr_values)
            print("Uniform budget", budgets)
            if total_aucpr == 0:
                pass
            else:
                budgets = [int(n * (coreset_num - sum(min_budgets.values())) / total_aucpr) if i not in min_budgets
                       else min_budgets[i] for i, n in enumerate(aucpr_values)]
                print("Initial budget", budgets)
                budgets = bin_allocate(coreset_num, strata_num, mode='aucpr', initial_budget=budgets)
            bin2budget = {bin_idx: budgets[i] for i, bin_idx in enumerate(non_empty_bins)}
        else:
            raise ValueError

        print('Using density sampling...')
        pools = []
        final_idxs = []
        # def get_coreset_for_bin(bins_to_accomplish, final_idxs):
        def get_coreset_for_bin(bin_idx):
            # while True:
            #     try:
            # bin_idx = bins_to_accomplish.get_nowait()
            sample_idxs = np.where(bins == bin_idx)[0]
            pools.append(sample_idxs)
            print("Starting process for label  ", bin_idx, "with %s samples" % len(sample_idxs))
            if len(sample_idxs) > 0:
                if bin2budget[bin_idx] > len(sample_idxs):
                    kcenters = np.random.permutation(sample_idxs.shape[0])
                else:
                    if args.sampling_mode == 'random':
                        rand_index = np.random.permutation(sample_idxs.shape[0])
                        kcenters = rand_index[:bin2budget[bin_idx]]
                    elif args.sampling_mode == 'kcenter':
                        sampling_method = kCenterGreedy(X=data_embeds[sample_idxs], y=None, seed=0)
                        kcenters = sampling_method.select_batch_(None, bin2budget[bin_idx])
                    elif args.sampling_mode == 'graph':
                        if sample_idxs.shape[0] <= args.n_neighbor:
                            kcenters = np.random.permutation(sample_idxs.shape[0])
                        else:
                            sampling_method = GraphDensitySampler(X=None if data_embeds is None else data_embeds[sample_idxs], y=None,
                                                                  gamma=args.gamma, seed=0,
                                                                  importance_scores=data_score[args.coreset_key][
                                                                      sample_idxs], args=args)
                                                                  # n_neighbor=args.n_neighbor,
                                                                  # graph_mode=args.graph_mode,
                                                                  # graph_sampling_mode=args.graph_sampling_mode,
                                                                  # precomputed_dists=args.precomputed_dists,
                                                                  # precomputed_neighbors=args.precomputed_neighbors
                                                                  # )
                            kcenters = sampling_method.select_batch_(bin2budget[bin_idx])
                    else:
                        raise ValueError
                kcenters = sample_idxs[kcenters]
            else:
                kcenters = []
            #             final_idxs.append(kcenters.tolist())
            #         else:
            #             continue
            #     except queue.Empty:
            #         break
            # return True
            return kcenters

        # number_of_processes = 1 if len(non_empty_bins) > 50 else 1
        # labels_to_accomplish = Queue()
        # processes = []
        #
        # manager = Manager()
        # final_idxs = manager.list()
        #
        # for bin_idx in non_empty_bins:
        #     labels_to_accomplish.put(bin_idx)
        # # creating processes
        # for w in range(number_of_processes):
        #     p = Process(target=get_coreset_for_bin, args=(labels_to_accomplish, final_idxs))
        #     processes.append(p)
        #     p.start()
        # # completing process
        # for p in processes:
        #     p.join()

        for bin_idx in non_empty_bins:
            kcenters = get_coreset_for_bin(bin_idx)
            final_idxs.append(kcenters.tolist())

        if args.aucpr:
            final_aucpr_values = []
            for pool, selected in zip(pools, final_idxs):
                non_coreset = list(set(pool.tolist()).difference(set(selected.tolist())))
                if len(non_coreset) == 0 or len(selected) == 0:
                    aucpr = 0
                else:
                    aucpr = get_aucpr(data_embeds[selected], data_embeds[non_coreset])
                final_aucpr_values.append(round(aucpr, 3))
            print("Final AUCpr values: ", final_aucpr_values)
            print("Final mean AUCpr: ", np.mean(final_aucpr_values))

        selected_idxs = []
        for idxs in final_idxs:
            selected_idxs.extend(idxs)
        random.shuffle(selected_idxs)
        if len(selected_idxs) < coreset_num:
            extra_sample_set = list(set(range(len(bins))).difference(set(selected_idxs)))
            selected_idxs = selected_idxs + random.sample(extra_sample_set, k=min(len(extra_sample_set), coreset_num-len(selected_idxs)))

        return selected_idxs[:coreset_num], None

    @staticmethod
    def random_selection(total_num, num):
        print('Random selection.')
        score_random_index = torch.randperm(total_num)

        return score_random_index[:int(num)]