import json
import collections
import codecs
import pickle


type_order = {'Procedural History': 1, 'Issue': 2, 'Service History':3, 'Evidential Support':4, 'Reasoning':4, 'Outcome':5}
valid_cases = "0739154 0825558 1018212 1432042 1209887 9933094 1022006 0330890 0425776 0516558 0730731 0947200 1409108 1610418 1446616 1639235 9702655 9737618 0125253 1721525 1109598 1540284 1525338 1448416 1706925 0728875 0919946 9627614 1452723 0902910 0611216 0116083 0406126 0911521 0935611 1037460 1414387 1454677 1640532 9810855 1706645 0523487 1712021 1235834 0725647 1030911 9921870 0836800 0632744 1313969 0918094 0730652 1535469 0928092 9604633 1615246 1513825 1417095 0534112 1338073 9923840 0839158 0330843 1624706 0717706 9522592 1712288 1141384 0519646 0114243 0723478 1629982 1038290 9909685 1715178 9834429 1139979 0927894 0110188 1202117 9414262 0308517 0528381 1036122 0917060 0120199 1705899 1631914 1218274 1039937 1236531 1635436".split()
# new_test_cases = "0704797 0204881 0206985 0218389 0207638 1339942 0825614 0924745 1454469 1631247 0403957 0121258 1727981 1445242 1203607 0017145 9932395 0206017 0010438 1640466".split()
dup_cases = '0330843 0632744 0717706 0928092 1313969 1513825 1535469 1615246 1624706 9522592 9921870'.split()
with open('20181129_bva_summ.json') as f:
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


def extract_outcome(filepath):
    lines = []
    with open(filepath, "rb") as f:
        for line in f:
            lines.append(line.decode("utf-8", errors="replace"))

    num_of_lines = len(lines)
    lines = [line.strip().lower() for line in lines]
    # print (len(lines))
    # print (lines)
    concat = " ".join(lines)
    concat = concat[int(len(concat) * 0.6):]
    concat2 = " ".join(lines)

    # print (concat)

    granted = "is granted" in concat
    remanded = "is remanded" in concat2 or "is REMANDED" in concat2
    denied = "is denied" in concat or "is dismissed" in concat

    # print ("%s %r %r %r" % (filepath, granted, remanded, denied))

    if (not (granted or remanded or denied)) and "is allowed" in concat:
        return 1

    """
        Several issues:
            1. some cases have both 'denied' and 'remanded', which mostly mean remanded
            2. some cases are none of the three, which are 'dismissed'. we treat them as 'remanded'
            3. some cases granted are described as 'allowed'

    """

    if granted:
        return 1
    if remanded:
        return 0
    if denied:
        return -1

    return 0


def extract_text():
    summary_lens = []
    case_len_map = collections.defaultdict(int)
    annotator_case_map = dict()
    not_valid_cnt = 0

    for doc in data['documents']:
        print()
        print(doc['name'])

        docid = doc['_id']  # 59d3e1b644a09d7f8e4763b8
        doc_name = doc['name']
        text = doc['plainText']
        annotator = None

        if len(annot_dict[docid]) == 0:
            print('not yet annotated')
            continue

        if doc_name[:-4] not in valid_cases and doc_name[:-4] not in dup_cases:
            print(doc_name[:-4], 'not a valid case')
            not_valid_cnt += 1
            continue

        sents = []
        tuples = []
        output = ''
        dup_case_flag = doc_name[:-4] in dup_cases
        case_outcome = extract_outcome("../../all-bva-decisions/" + doc_name)
        for annot in annot_dict[docid]:
            # print(annot.keys())
            sent = ' '.join(text[annot['start']: annot['end']].split())
            if sent in sents:
                continue
            if dup_case_flag and annot['owner'] == '5890e02d09f349251ffa58c8':
                continue

            sents.append(sent)
            output += '#%s#\n%s\n\n' % (types[annot['type']], sent)
            # output += '#%s#\n%s\n%s\n\n' % (types[annot['type']], sent, annot['owner'])
            tuples.append((types[annot['type']], sent))

            if annotator:
                try:
                    assert annot['owner'] == annotator
                except:
                    annotator = '5b831f365a9540537de3a49a'
            else:
                annotator = annot['owner']

        with codecs.open(dir + 'annot_' + doc_name, 'w', encoding='utf-8') as f:
            f.write(output)
        with codecs.open('ref_summaries/' + doc_name, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sents))
        # reorder by assumed type order
        with codecs.open('ordered_ref_summaries/' + doc_name, 'w', encoding='utf-8') as f:
            f.write('\n'.join([tup[1] for tup in sorted(tuples, key=lambda x: type_order[x[0]])]))

        with codecs.open('middle_ref_summaries/' + doc_name, 'w', encoding='utf-8') as f:
            f.write('\n'.join([tup[1] for tup in filter(lambda x: type_order[x[0]] == 4, tuples)]))

        summary_lens.append(len(sents))
        print(len(sents), 'sentences')
        case_len_map[doc_name[:-4]] = len(sents)
        print('annotator:', annotator)
        if annotator not in annotator_case_map:
            annotator_case_map[annotator] = collections.defaultdict(list)
        annotator_case_map[annotator][case_outcome].append(doc_name[:-4])

    print(summary_lens)
    print(min(summary_lens))
    print(max(summary_lens))
    print(annotator_case_map)
    with open('gold_summaries_len_map.pickle', 'wb') as handle:
        pickle.dump(case_len_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('annotator_case_map.pickle', 'wb') as handle:
        pickle.dump(annotator_case_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('gold_summaries_len_map.pickle', 'rb') as handle:
    #     b = pickle.load(handle)
    # assert case_len_map == b
    # print(not_valid_cnt)


if __name__ == '__main__':
    extract_text()
    pass

