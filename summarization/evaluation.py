from pythonrouge.pythonrouge import Pythonrouge
import os
import shutil
import collections
import statistics


def eval_rouge(hyp_dir, ref_dir):
    print(hyp_dir, 'vs', ref_dir)
    # create tmp dirs with common cases
    hyp_set = set(os.listdir(hyp_dir))
    ref_set = set(os.listdir(ref_dir))
    common_set = hyp_set.intersection(ref_set)
    # print(len(hyp_set))
    # print(len(ref_set))
    # print(len(common_set))

    tmp_hyp = './tmp_hyp'
    tmp_ref = './tmp_ref'

    if not os.path.isdir(tmp_hyp):
        os.mkdir(tmp_hyp)
    if not os.path.isdir(tmp_ref):
        os.mkdir(tmp_ref)

    # copy to tmp dirs
    for casefile in common_set:
        shutil.copyfile(os.path.join(hyp_dir, casefile), os.path.join(tmp_hyp, casefile))
        shutil.copyfile(os.path.join(ref_dir, casefile), os.path.join(tmp_ref, casefile))

    assert os.listdir(tmp_hyp) == os.listdir(tmp_ref)
    rouge = Pythonrouge(summary_file_exist=True,
                        peer_path=tmp_hyp, model_path=tmp_ref,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                        recall_only=True,
                        stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print(score)

    shutil.rmtree(tmp_hyp)
    shutil.rmtree(tmp_ref)


def eval_indiv_rouge(hyp_dir, ref_dir):
    print(hyp_dir, 'vs', ref_dir)
    # create tmp dirs with common cases
    hyp_set = set(os.listdir(hyp_dir))
    ref_set = set(os.listdir(ref_dir))
    common_set = hyp_set.intersection(ref_set)

    scores = collections.defaultdict(list)

    for casefile in common_set:
        tmp_hyp = './tmp_hyp'
        tmp_ref = './tmp_ref'
        os.mkdir(tmp_hyp)
        os.mkdir(tmp_ref)

        # copy to tmp dirs
        shutil.copyfile(os.path.join(hyp_dir, casefile), os.path.join(tmp_hyp, casefile))
        shutil.copyfile(os.path.join(ref_dir, casefile), os.path.join(tmp_ref, casefile))

        assert os.listdir(tmp_hyp) == os.listdir(tmp_ref)
        rouge = Pythonrouge(summary_file_exist=True,
                            peer_path=tmp_hyp, model_path=tmp_ref,
                            n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                            recall_only=True,
                            stemming=True, stopwords=True,
                            word_level=True, length_limit=True, length=50,
                            use_cf=False, cf=95, scoring_formula='average',
                            resampling=True, samples=1000, favor=True, p=0.5)
        score = rouge.calc_score()
        print(casefile, score)
        for key in score:
            scores[key].append(score[key])

        shutil.rmtree(tmp_hyp)
        shutil.rmtree(tmp_ref)

    print(["%s: mean %.3f std %.3f" % (key, round(sum(scores[key]) / len(scores[key]), 3), round(statistics.stdev(scores[key]), 3)) for key in scores])


if __name__ == '__main__':
    # hyp_dirs = ['./sample_case_summaries_60_middle', './sample_case_summaries_from_all_sentences_middle', 'sample_case_summaries_60_train_middle']
    # ref_dirs = ['./middle_ref_summaries']
    # hyp_dirs = ['./sample_case_summaries_60_test/random/']
    # ref_dirs = ['./annotators/5890e02d09f349251ffa58c8/ref_summaries']
    hyp_dirs = ['./sample_case_summaries_60_val_middle/']
    ref_dirs = ['./middle_ref_summaries']
    for hyp_dir in hyp_dirs:
        for ref_dir in ref_dirs:
            eval_indiv_rouge(hyp_dir, ref_dir)
    #
    #
    #
    # print ("full summary:")
    # hyp_dirs = ['./control_test/full/%.1f' % (x/10.) for x in range(1, 10)]
    # ref_dirs = ['./ordered_ref_summaries']
    # for hyp_dir in hyp_dirs:
    #     for ref_dir in ref_dirs:
    #         eval_rouge(hyp_dir, ref_dir)
    #
    # print ("\n\nmiddle part:")
    # hyp_dirs = ['./control_test_on_validation/middle/%.1f' % (x/10.) for x in range(1, 10)]
    # ref_dirs = ['./middle_ref_summaries']
    # # hyp_dirs = ['./sample_case_summaries_60_test_middle/5890e02d09f349251ffa58c8']
    # # ref_dirs = ['./annotators/5890e02d09f349251ffa58c8/middle_ref_summaries']
    # for hyp_dir in hyp_dirs:
    #     for ref_dir in ref_dirs:
    #         eval_rouge(hyp_dir, ref_dir)

