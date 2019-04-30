import os, json, codecs, random
from shutil import copy
from template_sents_extraction import find_issue


def PTSD_in_issue(issue):
	return "PTSD" in issue or \
		   "posttraumatic stress disorder" in issue.lower() or \
		   "post-traumatic stress disorder" in issue.lower()

def not_multiple(issue):
	return not ("1." in issue and "2." in issue)


# def not_secondary(issue):
# 	return "secondary" not in issue.lower()


# def service_connection_entitle(issue):
# 	return "entitlement to service connection" in issue.lower()


def service_connection(issue):
	return "service connection" in issue.lower()


def not_reopen(issue):
	return "reopen" not in issue and "re-open" not in issue


def not_incr_eval(issue):
	return "percent" not in issue and "per cent" not in issue and "per-cent" not in issue


def special_case_exact_match(issue):
	issue = issue.lower()
	return issue.startswith("the issue is service connection for post-traumatic stress disorder") or \
		   issue.startswith("the issue is service connection for posttraumatic stress disorder")




BASE_DIR = "../../all-bva-decisions"
output = open("single_issue_PTSD_Nov_28.txt", "w")
print (len(os.listdir(BASE_DIR)))
for i, case in enumerate(os.listdir(BASE_DIR)):

	if (i+1) % 10000 == 0:
		print (i+1)

	f = codecs.open(os.path.join(BASE_DIR, case), encoding="utf-8", errors="replace")
	lines = f.readlines()
	if any(l.startswith("THE ISSUES") for l in lines):
		continue

	issue = find_issue(case).strip()
	criteria = [
		PTSD_in_issue,
		not_multiple,
		service_connection,
		not_reopen,
		not_incr_eval,
	]

	if special_case_exact_match(issue) or all(criterion(issue) for criterion in criteria):
		output.write(case + "\n")

	f.close()

	# check THE ISSUES
	# f = codecs.open(os.path.join(BASE_DIR, case), encoding="utf-8", errors="replace")
	# lines = f.readlines()
	# if any(l.startswith("THE ISSUES") for l in lines):
	# 	issue = find_issue(case)
	# 	if "PTSD" in issue or "traumatic" in issue.lower():
	# 		print (case, "\t", issue)
	# 		pause = input("")
	# f.close()


output.close()
