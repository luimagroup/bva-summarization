import json
import collections
import codecs
import pickle
import os


type_order = {'Procedural History': 1, 'Issue': 2, 'Service History':3, 'Evidential Support':4, 'Reasoning':4, 'Outcome':5}
new_test_cases = "0704797 0204881 0206985 0218389 0207638 1339942 0825614 0924745 1454469 1631247 0403957 0121258 1727981 1445242 1203607 0017145 9932395 0206017 0010438 1640466".split()
with open('20190125_bva.json') as f:
    data = json.load(f)
# print(data['annotations'])

types = {}
for each in data['types']:
    types[each['_id']] = each['name']
print(types)

dir = 'raw_gold_summaries/'
annot_dict = collections.defaultdict(list)
for annot in data['annotations']:
    annot_dict[annot['document']].append(annot)

for doc in annot_dict:
    annot_dict[doc] = sorted(annot_dict[doc], key=lambda annot: annot['start'])

print("total doc num", len(data['documents']))
print("annotated doc num", len(annot_dict))


def extract_text():
    case_len_map = dict()
    not_valid_cnt = 0

    for doc in data['documents']:
        print()
        print(doc['name'])
        annotators_sents = dict()
        annotators_tuples = dict()
        annotators_outputs = dict()

        docid = doc['_id']  # 59d3e1b644a09d7f8e4763b8
        doc_name = doc['name']
        text = doc['plainText']

        if len(annot_dict[docid]) == 0:
            print('not yet annotated')
            continue

        if doc_name[:-4] not in new_test_cases:
            print(doc_name[:-4], 'not a valid case')
            not_valid_cnt += 1
            continue

        for annot in annot_dict[docid]:
            curr_annotator = annot['owner']
            if curr_annotator not in annotators_sents:
                annotators_sents[curr_annotator] = []
                annotators_tuples[curr_annotator] = []
                annotators_outputs[curr_annotator] = ''

            sent = ' '.join(text[annot['start']: annot['end']].split())
            if sent in annotators_sents[curr_annotator]:
                continue

            annotators_sents[curr_annotator].append(sent)
            annotators_outputs[curr_annotator] += '#%s#\n%s\n\n' % (types[annot['type']], sent)
            annotators_tuples[curr_annotator].append((types[annot['type']], sent))

        for each_annotator in annotators_sents:
            prefix = './annotators/' + each_annotator + '/'
            if not os.path.exists(prefix + dir):
                os.makedirs(prefix + dir)
            if not os.path.exists(prefix + 'ref_summaries/'):
                os.makedirs(prefix + 'ref_summaries/')
            if not os.path.exists(prefix + 'ordered_ref_summaries/'):
                os.makedirs(prefix + 'ordered_ref_summaries/')
            if not os.path.exists(prefix + 'middle_ref_summaries/'):
                os.makedirs(prefix + 'middle_ref_summaries/')

            with codecs.open(prefix + dir + 'annot_' + doc_name, 'w', encoding='utf-8') as f:
                f.write(annotators_outputs[each_annotator])
            with codecs.open(prefix + 'ref_summaries/' + doc_name, 'w', encoding='utf-8') as f:
                f.write('\n'.join(annotators_sents[each_annotator]))
            # reorder by assumed type order
            with codecs.open(prefix + 'ordered_ref_summaries/' + doc_name, 'w', encoding='utf-8') as f:
                f.write('\n'.join([tup[1] for tup in sorted(annotators_tuples[each_annotator],
                                                            key=lambda x: type_order[x[0]])]))

            with codecs.open(prefix + 'middle_ref_summaries/' + doc_name, 'w', encoding='utf-8') as f:
                f.write('\n'.join([tup[1] for tup in filter(lambda x: type_order[x[0]] == 4,
                                                            annotators_tuples[each_annotator])]))

            if each_annotator not in case_len_map:
                case_len_map[each_annotator] = collections.defaultdict(int)
            case_len_map[each_annotator][doc_name[:-4]] = len(annotators_sents[each_annotator])

    with open('gold_summaries_len_map_per_annotator.pickle', 'wb') as handle:
        pickle.dump(case_len_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    case_len_map_avg = collections.defaultdict(int)
    for each_annotator in case_len_map:
        for each_case in case_len_map[each_annotator]:
            case_len_map_avg[each_case] += case_len_map[each_annotator][each_case]
    for each_case in case_len_map_avg:
        case_len_map_avg[each_case] = int(round(case_len_map_avg[each_case] / 4.0))

    with open('gold_summaries_len_map_avg.pickle', 'wb') as handle:
        pickle.dump(case_len_map_avg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(not_valid_cnt)


if __name__ == '__main__':
    extract_text()
    pass

