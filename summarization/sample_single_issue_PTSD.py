import os, json, codecs, random
from shutil import copy

BASE_DIR = "../../all-bva-decisions"

# g = open("PTSD_all_issue_sentence.json", "r")
# issues = json.loads(g.readline())
# g.close()

f = open("../single_issue/PTSD_trainCases.txt", "r")
collection = []
for line in f.readlines():
	line = line.strip()
	# if "PTSD" not in issues[line]:
	# 	continue
	# case = codecs.open(os.path.join(BASE_DIR, line), encoding="utf-8", errors="replace")
	# ls = case.readlines()
	# if any(l.startswith("THE ISSUES") for l in ls):
	# 	continue

	collection.append(line)

sample_issues = random.sample(collection, 200)
# output = open("../single_issue/real_PTSD_trainCases.txt", "w")
# for issue in collection:
# 	output.write(issue + "\n")

# output.close()
output = open("./single_issue_PTSD_sample/sample_cases.txt", "w")
for issue in sample_issues:
	copy(os.path.join(BASE_DIR, issue), "./single_issue_PTSD_sample/")
	output.write(issue + "\n")

output.close()



