import torch
import pdb
import numpy as np
from collections import defaultdict, namedtuple
import pickle
import json
# create named tuple

OneInstance = namedtuple('NamedTuple', ['responses', 'answers', 'scores', 'ppls', "question", "gold_answer"])

class StatsCollector:
    def __init__(self, score_model = None):
        self.instance_list = []
        self.score_model = score_model

    def add_batch(self, rollouts, sample_rounds): # add a batch of rollouts

        current_batch_prompts = []
        current_batch_answers = []
        for i in range(len(rollouts['query/text'])):
            current_batch_prompts.extend([rollouts['query/text'][i]] * sample_rounds)
            current_batch_answers.extend([rollouts['query/answer'][i]] * sample_rounds)
        current_batch_responses_text = rollouts['response/text']
        current_batch_responses_answer = rollouts['response/answer']

        batch_size = len(current_batch_prompts) // sample_rounds
        batch_scores = self.score_model.get_reward(current_batch_prompts, current_batch_responses_answer, answers = current_batch_answers)

        multiround_batch_scores = torch.LongTensor(batch_scores).view(batch_size, sample_rounds)
        if 'response/normalized_sentence_probs' in rollouts:
            probs = rollouts['response/normalized_sentence_probs']
            probs = [probs[i * sample_rounds: (i + 1) * sample_rounds] for i in range(batch_size)]
        else:
            # put zeros 
            probs = [[0] * sample_rounds] * batch_size
        

        for i in range(batch_size):
            instance = OneInstance(
                current_batch_responses_text[i * sample_rounds: (i + 1) * sample_rounds], 
                current_batch_responses_answer[i * sample_rounds: (i + 1) * sample_rounds], 
                multiround_batch_scores[i].tolist(), 
                [np.exp(-j) for j in probs[i]],
                rollouts['query/text'][i],
                rollouts['query/answer'][i])
            self.instance_list.append(instance)

    def dump_to_file(self, filename):
        new_list = [i._asdict() for i in self.instance_list]
        with open(filename, 'w') as f:
            json.dump(new_list, f)
    
    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            content = json.load(f)
            for i in content:
                self.instance_list.append(OneInstance(**i))

    def bin_and_calculate_stats(self, all_ppls, scores, bin_num):
        max_ppl = max(all_ppls)
        min_ppl = min(all_ppls)
        ppl_range = max_ppl - min_ppl
        bin_size = ppl_range / bin_num
        bin_edges = [min_ppl + i * bin_size for i in range(bin_num + 1)] + [100]

        bin_indexes = np.digitize(all_ppls, bin_edges)
        acc_bins = defaultdict(list)
        index_bins = defaultdict(list)
        for index in range(len(bin_indexes)):
            acc_bins[bin_indexes[index]].append(scores[index])
            index_bins[bin_indexes[index]].append(index)

        # print by sorted key of acc_bins
        for index in sorted(acc_bins.keys()):
            print(index, bin_edges[index], np.std(acc_bins[index]), len(acc_bins[index]))
        
        print()
        for index in sorted(acc_bins.keys()):
            print(np.mean(acc_bins[index]))
        print()
        for index in sorted(acc_bins.keys()):
            # print bin_edges[index] upto one decimal point
            print("{:.1f}".format(bin_edges[index]))
        
        print()
        for index in sorted(acc_bins.keys()):
            # print bin_edges[index] upto one decimal point
            print("{:.2f}".format(bin_edges[index]))

        return index_bins

    def report_ppl_correlation(self):
        # calculate range
        all_ppls = []
        all_scores = []
        for i in self.instance_list:  
            all_ppls.extend(i.ppls)
            all_scores.extend(i.scores)

        index_bins = self.bin_and_calculate_stats(all_ppls, all_scores, 20)

        print("The first bing")
        first_bin_index = sorted(list(index_bins.keys()))[0]
        first_bin_indexes = index_bins[1] + index_bins[2]
        first_bin_scores = [all_scores[i] for i in first_bin_indexes]
        first_bin_ppls = [all_ppls[i] for i in first_bin_indexes]
        self.bin_and_calculate_stats(first_bin_ppls, first_bin_scores, 20)


        print("The middlle bing")
        first_bin_index = sorted(list(index_bins.keys()))[0]
        first_bin_indexes = index_bins[1] + index_bins[2] + index_bins[3] + index_bins[4] + index_bins[5] + index_bins[6] + index_bins[7] + index_bins[8] + index_bins[9] + index_bins[10] + index_bins[11]
        first_bin_scores = [all_scores[i] for i in first_bin_indexes]
        first_bin_ppls = [all_ppls[i] for i in first_bin_indexes]
        self.bin_and_calculate_stats(first_bin_ppls, first_bin_scores, 10)


if __name__ == '__main__':
    import os
    dir_eval = os.environ['DIR_EVAL']
    file_name = "OUTPUTS/{}/stats.pickle".format(dir_eval)
    stats_collector = StatsCollector()
    stats_collector.load_from_file(file_name)
    stats_collector.report_ppl_correlation()

