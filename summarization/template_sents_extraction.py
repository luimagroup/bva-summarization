import os, codecs, json, re, us, pickle

BASE_DIR = "../../all-bva-decisions"
TRAIN_CASES = "../single_issue/PTSD_trainCases.txt"
ALL_CASES = "../single_issue/single_issue_PTSD_all.txt"
STATES = [state.lower() for state in us.states.mapping("abbr", "name").values()]

outcome_file = open("../sentence_encoder/PTSD_trainCases_outcome.txt", "r")
# outcome_file = open("../sentence_encoder/sample_case_outcome.txt", "r")
num2outcome = {-1: ["denied", "dismissed"], 0: ["remanded"], 1: ["granted"]}
outcomes = [line.strip().split() for line in outcome_file.readlines()]
OUTCOME_DICT = {x: num2outcome[int(y)] for x, y in outcomes}
outcome_file.close()


def extract_issue():
    """
    Extract issue for all single-issue PTSD training cases.
    """
    train_cases_f = open(ALL_CASES, "r")
    train_cases = [line.strip() for line in train_cases_f.readlines()]
    train_cases_f.close()
    issues = map(find_issue, train_cases)
    result = open("PTSD_all_issue_sentence.json", "w")
    result.write(json.dumps({case: issue for case, issue in zip(train_cases, issues)}))
    

def find_issue(case):
    """
    Find the issue sentence for a given case.
    """

    if ".txt" not in case:
        case += ".txt"

    f = codecs.open(os.path.join(BASE_DIR, case), encoding="utf-8", errors="replace")
    issue, switch = "", False
    for line in f.readlines():
        if line.startswith("THE ISSUE"):
            switch = True
        elif switch and line.strip() != "" and line == line.upper():
            break
        elif switch:
            issue += line

    return "The issue is " + " ".join(issue.split())


def extract_appeals():
    """
    Extract appeal sentence fr all single-issue PTSD training cases.
    """
    train_cases_f = open(ALL_CASES, "r")
    train_cases = [line.strip() for line in train_cases_f.readlines()]
    train_cases_f.close()
    appeals = map(find_appeal, train_cases)
    result = open("PTSD_all_appeal_sentence.json", "w")
    result.write(json.dumps({case: appeal for case, appeal in zip(train_cases, appeals)}))


def find_appeal(case):
    """
    Find the appeal introduction sentence for a given case.
    If not found, empty string is returned.
    """

    if ".txt" not in case:
        case += ".txt"

    f = codecs.open(os.path.join(BASE_DIR, case), encoding="utf-8", errors="replace")
    lines = [line.strip() for line in f.readlines()]
    appeal = ""
    for i, line in enumerate(lines):
        if "appeal from" in line.lower():
            appeal = " ".join(lines[i:i+3])
            return " ".join(appeal.split()) + "."

        if "before the board" in line.lower():
            for j in range(i, i+4):
                appeal += (" " + lines[j])
                for state in STATES:
                    if state in lines[j].lower():
                        break
            appeal = " ".join(appeal.split())
            st = appeal.lower().find(state)
            ed = st + len(state)
            return appeal[:ed] + "."

    return appeal



def find_conclusion(case, sents):
    """
    Find conclusion sentence from selected predictive sentences.
    :param case: case id
    :param sents: list of sentences to be selected from
    :return a concluding sentence
    """

    if ".txt" not in case:
        case += ".txt"

    outcome = OUTCOME_DICT[case]
    keywords = ["is " + oc for oc in outcome]
    conclusion = ""
    for sent in sents:
        if any(kw in sent.lower() for kw in keywords):
            conclusion = sent
            break

    return conclusion if conclusion else "This case is " + outcome[0]



def find_service_time(case):

    """
        If not found return None.

    """

    keywords = [
        "active duty",
        "active service",
        "military service",
        "had service",
        "air service",
        "honorable service"
    ]

    # patterns = [
    #     "from [a-zA-Z]+\s+\d{4} to [a-zA-Z]+\s+\d{4}",
    #     "from [a-zA-Z]+\s+\d+,?\s+\d{4} to [a-zA-Z]+\s+\d+,?\s+\d{4}",
    #     "from [a-zA-Z]+\s+\d{4} until [a-zA-Z]+\s+\d{4}",
    #     "from [a-zA-Z]+\s+\d+,?\s+\d{4} until [a-zA-Z]+\s+\d+,?\s+\d{4}",
    # ]

    sents = segmented_sents(case if ".txt" not in case else case[:-4])
    for sent in sents:
        if any(kw in sent for kw in keywords):
            sent = sent.replace('\r', '')
            sent = sent.replace('\n', '')
            return sent

    return None




###### Util ######

def segmented_sents(caseid):

    name = caseid + '.pkl'
    # TODO: Change to relative path
    # path = '../bva-segments/'
    path = '../../bva-segments/'

    file = open(path+name, 'rb')
    sent_list = pickle.load(file)

    with open('../../all-bva-decisions/' + caseid + '.txt', 'rb') as file:
        case_str = file.read().decode("ISO8859-1")

    para_i = 0
    sent2para = []
    AFTER_INTRO = False

    # list of list of tuples: para sent start/end pair
    para_sent_pairs = []
    sent_pairs = []

    # sent string list
    sent_strings = []

    # para2idx
    para2idx = {}
    idx2para = {}
    para_string = []

    for item in sent_list:
        # print(item)
        sent_string = (case_str[item[0]:item[1]])
        # print(sent_string)
        ending = case_str[item[1] - 4:item[1] + 16]  # hardcoded to be most robust

        if AFTER_INTRO:
            sent2para.append(para_i)
            sent_strings.append(sent_string)

            # check ending and see if it's end of paragraph
            if ending.count('\r\n') >= 2:
                para_sent_pairs.append(sent_pairs)
                para = '  '.join(para_string)
                para2idx[para[:50]] = para_i
                idx2para[para_i] = para[:50]
                para_i += 1

                sent_pairs = []

        if 'INTRODUCTION' in sent_string:
            AFTER_INTRO = True

    return sent_strings

###### testing ######

def test_issue():
    """
    Testing results shows that "THE ISSUE(S)" must appear in the case.
    """

    train_cases_f = open(TRAIN_CASES, "r")
    train_cases = [line.strip() for line in train_cases_f.readlines()]
    train_cases_f.close()

    for case in train_cases:
        f = codecs.open(os.path.join(BASE_DIR, case), encoding="utf-8", errors="replace")
        lines = [line.strip() for line in f.readlines()]
        if "THE ISSUE" not in lines and "THE ISSUES" not in lines:
            print (case)

        f.close()


def test_issue_extraction_result():

    """
    Show extracting result in pprint fashion.
    """

    from pprint import pprint
    with open("PTSD_all_issue_sentence.json") as f:
        result = json.load(f)

    pprint(result)


def test_appeal():
    
    train_cases_f = open(TRAIN_CASES, "r")
    train_cases = [line.strip() for line in train_cases_f.readlines()]
    train_cases_f.close()

    for case in train_cases:
        f = codecs.open(os.path.join(BASE_DIR, case), encoding="utf-8", errors="replace")
        lines = [line.strip() for line in f.readlines()]
        # if not any(line.startswith("This matter came before") or line.startswith("This appeal came before") for line in lines):
        #   print (case)
        # pattern = re.compile("(This [A-Za-z]+ came before [A-Za-z]+)|Department of Veterans Affairs [A-Za-z,]*")
        if not any("appeal from" in line.lower() or
                   "before the board" in line.lower() or
                   "received from the" in line.lower() for line in lines):
            print (case)


def test_appeal_extraction_result():

    """
    Show extracting result in pprint fashion.
    """

    from pprint import pprint
    with open("PTSD_trainingCases_appeal_sentence.json") as f:
        result = json.load(f)

    pprint(result)


def test_service_time():

    train_cases_f = open(TRAIN_CASES, "r")
    train_cases = [line.strip() for line in train_cases_f.readlines()]
    train_cases_f.close()

    print (len(train_cases))
    num = 0
    for case in train_cases:
        f = codecs.open(os.path.join(BASE_DIR, case), encoding='utf-8', errors="replace")
        lines = [line.strip() for line in f.readlines()]

        # switch, printed = False, False
        # for line in lines:
        #     if line.startswith("INTRODUCTION"):
        #         switch = True
        #     elif switch and line:
        #         print (line)
        #         printed = True
        #     elif printed:
        #         break

        # if not printed:
        #     print (case, " no intro section.")


        text = " ".join(lines)
        # pattern = "from\s+[a-zA-Z]+\s+\d{4}\s+to\s+a-zA-Z]+\s+\d{4}"
        pattern = "from [a-zA-Z]+\s+\d{4} to [a-zA-Z]+\s+\d{4}"
        pattern2 = "from [a-zA-Z]+\s+\d+,?\s+\d{4} to [a-zA-Z]+\s+\d+,?\s+\d{4}"
        pattern3 = "from [a-zA-Z]+\s+\d{4} until [a-zA-Z]+\s+\d{4}"
        pattern4 = "from [a-zA-Z]+\s+\d+,?\s+\d{4} until [a-zA-Z]+\s+\d+,?\s+\d{4}"
        level1 = re.findall(pattern, text) + re.findall(pattern3, text)
        level2 = re.findall(pattern4, text) + re.findall(pattern2, text)
        if len(level1) > 2 or (not level1 and len(level2) > 2):
            print (case, len(level1), len(level2))
            num += 1

        # if case == "0819208.txt":
            # print (temp)
            # print (text)



        # keywords = [
        #     "active duty",
        #     "active service",
        #     "military service",
        #     "had service",
        #     "air service",
        #     "honorable service"
        # ]
        # if not temp and all(kw not in text for kw in keywords):
        #     print (case, OUTCOME_DICT[case])
            


    print(num)




if __name__ == '__main__':
    # test_issue()
    # print (find_issue("0023497.txt"))
    # extract_issue()
    # test_issue_extraction_result()
    # test_appeal()
    # print (find_appeal("0932095.txt"))
    # extract_appeals()
    # test_appeal_extraction_result()
    # test_service_time()
    for case in os.listdir("./gold_summaries"):
        print(case, find_service_time(case[6:]))
        pause = input("")



