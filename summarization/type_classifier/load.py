import os
import re
import codecs
from pprint import pprint


def read_files(cases_names, documents_dir='../../new_annotated_casetext'):
    """
     Read annotated case files
    :param documents_dir: directory of case files
    :return: each annotated sentence and its type
    """
    annotated_lines = []
    type_pattern = re.compile("#[\s\S]*#")
    for case_name in cases_names:
        with codecs.open(os.path.join(documents_dir, case_name.strip()), 'rb', encoding='utf-8') as case:
            lines = []
            line_type = ''
            for line in case:
                # line = line.decode('utf-8', errors='replace').strip()
                line = line.strip()
                # If starts to meet a new line
                if line == '':
                    if len(lines) != 0:
                        annotated_line = ' '.join(lines)
                        annotated_lines.append((annotated_line, line_type))
                    lines = []
                    line_type = ''
                elif re.match(type_pattern, line):
                    line_type = line.strip('#')
                else:
                    lines.append(line)

    return annotated_lines


'''
def read_files(cases_names, documents_dir='../annotated_casetext'):
    """
     Read annotated case files
    :param documents_dir: directory of case files
    :return: each annotated sentence and its type
    """
    type_map = {'FindingSentence': 'EvidenceBasedFinding',
                'EvidenceSentence': 'Evidence',
                'EvidenceBasedReasoningSentence': 'EvidenceBasedReasoning',
                'CitationSentence': 'Citation',
                'LegalRuleSentence': 'LegalRule'}
    types = ['Citation', 'LegalRule', 'LegalPolicy', 'PolicyBasedReasoning',
             'ConclusionOfLaw', 'EvidenceBasedFinding', 'EvidenceBasedReasoning',
             'Evidence', 'Procedure', 'Header']
    type_set = set(types)

    annotated_lines = []
    type_pattern = re.compile("#[\s\S]*#")
    for case_name in cases_names:
        with open(os.path.join(documents_dir, case_name.strip()), 'rb') as case:
            lines = []
            line_type = ''
            for line in case:
                line = line.decode('utf-8', errors='replace').strip()
                # If starts to meet a new line
                if line == '':
                    if len(lines) != 0 and line_type in type_set:
                        annotated_line = ' '.join(lines)
                        annotated_lines.append((annotated_line, line_type))
                    lines = []
                    line_type = ''
                elif re.match(type_pattern, line):
                    line_type = line.strip('#')
                    if line_type in type_map:
                        line_type = type_map[line_type]
                else:
                    lines.append(line)

    return annotated_lines
'''
