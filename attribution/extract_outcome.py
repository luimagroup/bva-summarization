import sys, os
from collections import defaultdict


def extract_outcome(filepath):
    lines = []
    with open(filepath, "rb") as f:
        for line in f:
            lines.append(line.decode("utf-8", errors="replace"))

    num_of_lines = len(lines)
    lines = [line.strip().lower() for line in lines]
    #print (len(lines))
    #print (lines)
    concat = " ".join(lines)
    concat = concat[int(len(concat)*0.6):]
    concat2 = " ".join(lines)
    
    #print (concat)

    granted = "is granted" in concat
    remanded = "is remanded" in concat2 or "is REMANDED" in concat2
    denied = "is denied" in concat or "is dismissed" in concat

    #print ("%s %r %r %r" % (filepath, granted, remanded, denied))

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


# Test
if __name__ == '__main__':

    option = sys.argv[1]
    BASE_DIR = "../../all-bva-decisions/"

    outputf = "PTSD_%sCases_outcome.txt" % option
    if option == "all":
        outputf = "PTSD_all_outcome.txt"

    output = open(outputf, "w")

    fn = "../single_issue/PTSD_%sCases.txt" % option
    if option == "all":
        fn = "../single_issue/single_issue_PTSD_all_old.txt"
    f = open(fn, "r")


    c = defaultdict(int)

    for case in f.readlines():
        outcome = extract_outcome(os.path.join(BASE_DIR, case.strip()))
        output.write("%s %d\n" % (case.strip(), outcome))
        c[outcome] += 1

    f.close()
    output.close()
    print (c)

