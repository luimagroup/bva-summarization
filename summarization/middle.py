import os, codecs

if __name__ == "__main__":
    DIR = "./sample_case_summaries_60_test/avg/0.8"
    TARGET = "./sample_case_summaries_60_test_middle/avg/0.8"
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)

    for case in os.listdir(DIR):
        f = codecs.open(os.path.join(DIR, case), encoding="utf-8", errors="replace")
        lines = f.readlines()[3:-1]
        wf = codecs.open(os.path.join(TARGET, case), "w", encoding="utf-8")
        for l in lines:
            wf.write(l)
