import pickle
import numpy as np
from get_attribution import generate_testcases, get_top_sentence
import os, sys
import codecs


def backup_data_dir(data_dir):
    dir_name = data_dir[data_dir.rfind('/') + 1:]
    tmp_dir = 'tmp' + dir_name[0].upper() + dir_name[1:]
    import shutil, os
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    for f in os.listdir(data_dir):
        shutil.copy(os.path.join(data_dir, f), tmp_dir)
    return tmp_dir


def mask_out(path, caseid, sent_indices):
    f = open(os.path.join(path, caseid + '.pickle'), 'rb')
    x, y = pickle.load(f)
    for sent_idx in sent_indices:
        x[sent_idx, :] = np.zeros(x.shape[1])

    # write the updated embeddings x
    handle = open(os.path.join(path, caseid + '.pickle'), 'wb')
    pickle.dump([x, y], handle, protocol=2)


def dir_migration(path):
    dir_name = path[path.rfind('/') + 1:]
    tmp_dir = 'tmp' + dir_name[0].upper() + dir_name[1:]
    if os.path.exists(tmp_dir):
        path = tmp_dir
    else:
        # Copy original encoding data to a tmp directory
        backup_data_dir(path)
        path = tmp_dir
    return path


def run(round_num, mask_val_flag, verbose=1):
    model_path = 'weights.best.hdf5'
    train_data_path = dir_migration('../sentence_encoder/trainEmbeddings')
    # Create directory for per-case masked sentences if necessary
    train_case_dir = 'per_case_masked_sentences_train'
    if not os.path.exists(train_case_dir):
        os.mkdir(train_case_dir)

    # If also need to mask validation dataset cases
    if mask_val_flag:
        val_data_path = dir_migration('../sentence_encoder/valEmbeddings')
        val_case_dir = 'per_case_masked_sentences_val'
        if not os.path.exists(val_case_dir):
            os.mkdir(val_case_dir)
    else:
        val_case_dir = ''
        val_data_path = ''

    # For training dataset
    data_num = len(os.listdir(train_data_path))
    # Get top sentences
    results = get_top_sentence(model_path, generate_testcases(train_data_path, limit=data_num), k=1)
    round_log_name = 'mask_round' + round_num + '_train_sentences.txt'
    with codecs.open(round_log_name, 'w', 'utf-8') as f:
        for i, result in enumerate(results):
            case_id = result['caseid']
            case_file = codecs.open(os.path.join(train_case_dir, case_id + '.txt'), 'a', 'utf-8')
            f.write("%d: case %s, outcome:%d\n" % (i, case_id, result['outcome']))
            if verbose == 1:
                print("train case %d" % i)
            sentences = result['positive']
            sent_indices = []
            for each in sentences:
                f.write(each[2] + "\n\n")
                case_file.write("round " + round_num + ", index " + str(each[0]) + ": \n" + each[2] + "\n\n")
                sent_indices.append(each[0])
            case_file.close()
            mask_out(train_data_path, case_id, sent_indices)

    # For validation dataset
    if mask_val_flag:
        data_num = len(os.listdir(val_data_path))
        # Get top sentences
        results = get_top_sentence(model_path, generate_testcases(val_data_path, limit=data_num), k=1)
        round_log_name = 'mask_round' + round_num + '_val_sentences.txt'
        with codecs.open(round_log_name, 'w', 'utf-8') as f:
            for i, result in enumerate(results):
                case_id = result['caseid']
                case_file = codecs.open(os.path.join(val_case_dir, case_id + '.txt'), 'a', 'utf-8')
                f.write("%d: case %s\n" % (i, case_id))
                if verbose == 1:
                    print("val case %d" % i)
                sentences = result['positive']
                sent_indices = []
                for each in sentences:
                    f.write(each[2] + "\n\n")
                    case_file.write("round " + round_num + ", index " + str(each[0]) + ": \n" + each[2] + "\n\n")
                    sent_indices.append(each[0])
                case_file.close()
                mask_out(val_data_path, case_id, sent_indices)


def run_validation(round_num, verbose=1):
    model_path = 'weights.best.hdf5'

    val_data_path = dir_migration('../sentence_encoder/valEmbeddings')
    val_case_dir = 'per_case_masked_sentences_val'
    if not os.path.exists(val_case_dir):
        os.mkdir(val_case_dir)

    # For validation dataset
    data_num = len(os.listdir(val_data_path))
    # Get top sentences
    results = get_top_sentence(model_path, generate_testcases(val_data_path, limit=data_num), k=1)
    round_log_name = 'mask_round' + round_num + '_val_sentences.txt'
    with codecs.open(round_log_name, 'w', 'utf-8') as f:
        for i, result in enumerate(results):
            case_id = result['caseid']
            case_file = codecs.open(os.path.join(val_case_dir, case_id + '.txt'), 'a', 'utf-8')
            f.write("%d: case %s\n" % (i, case_id))
            if verbose == 1:
                print("val case %d" % i)
            sentences = result['positive']
            sent_indices = []
            for each in sentences:
                f.write(each[2] + "\n\n")
                case_file.write("round " + round_num + ", index " + str(each[0]) + ": \n" + each[2] + "\n\n")
                sent_indices.append(each[0])
            case_file.close()
            mask_out(val_data_path, case_id, sent_indices)


def check_top_k_sentences(k=40, verbose=1):
    """
    Check top k masked sentences for the first round
    """
    model_path = 'weights.best.hdf5'
    data_path = dir_migration('../sentence_encoder/trainEmbeddings')
    # Create directory for per-case masked sentences if necessary
    case_dir = 'top_' + str(k) + '_sentences_first_round'
    if not os.path.exists(case_dir):
        os.mkdir(case_dir)

    # For training dataset
    data_num = len(os.listdir(data_path))
    # Get top sentences
    results = get_top_sentence(model_path, generate_testcases(data_path, limit=data_num), k=k)
    for i, result in enumerate(results):
        case_id = result['caseid']
        case_file = codecs.open(os.path.join(case_dir, case_id + '.txt'), 'a', 'utf-8')
        if verbose == 1:
            print("train case %d" % i)
        sentences = result['positive']
        for idx, each in enumerate(sentences):
            case_file.write("top " + str(idx) + ", index " + str(each[0]) + ": \n" + each[2] + "\n\n")
        case_file.close()


def mask_on_subset(cases, max_round=40, verbose=1):
    """
    Use generated models to mask given cases for the given number of rounds
    """
    cases = set(cases)
    case_dir = "sample_case_masked_sentences"
    if not os.path.exists(case_dir):
        os.mkdir(case_dir)

    for round_num in range(max_round):
        model_path = os.path.join("./new_models/round" + str(round_num), "weights.best.hdf5")
        data_path = dir_migration('../sentence_encoder/trainEmbeddings')
        data_num = len(os.listdir(data_path))
        # Get top sentences
        results = get_top_sentence(model_path, generate_testcases(data_path, limit=data_num), k=1)
        for i, result in enumerate(results):
            case_id = result['caseid']
            case_name = case_id + '.txt'
            if case_name not in cases:
                print(case_name)
                continue

            case_file = codecs.open(os.path.join(case_dir, case_id + '.txt'), 'a', 'utf-8')
            if verbose == 1:
                print("case %d" % i)
            sentences = result['positive']
            sent_indices = []
            for each in sentences:
                case_file.write("round " + str(round_num) + ", index " + str(each[0]) + ": \n" + each[2] + "\n\n")
                sent_indices.append(each[0])
            case_file.close()
            mask_out(data_path, case_id, sent_indices)
        print("========================================")
        print("Finish round " + str(round_num))


if __name__ == '__main__':
    # mask_val = True if sys.argv[2] == 'True' else False
    # run(sys.argv[1], mask_val)
    # check_top_k_sentences()
    cases = []
    with open("sample_cases.txt", "r") as f:
        for each in f.readlines():
            each = each.strip()
            if each.endswith(".txt"):
                cases.append(each)
    print(len(cases))
    mask_on_subset(cases)
