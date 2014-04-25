#!/usr/local/bin/python
"""Transforms an HTK MLF file with phones into a text file.

Usage:
    mlf_to_text.py [-] [--forcealigned] [--timitfoldings]
    mlf_to_text.py < file.mlf > file.txt
    mlf_to_text.py --forcealigned < aligned_file.mlf > file.txt
    assumes an MLF with lines: TIME_START TIME_END PHONE
    & for a forced aligned MLF: TIME_START TIME_END PHONE_STATE LIK [PHONE]

Options:
    -h --help        Show this screen.
    --version        Show version.
    --forcealigned   Use a forced aligned MLF (longer columns) 
    --timitfoldings  Uses the timit_foldings.json file to regroup phones

"""

import sys, json
from docopt import docopt

def parse(l, forcealigned=False, foldings={}):
    res = []
    for line in l:
        if not len(line):
            continue
        t = line.rstrip('\n').split()
        if t[0].isdigit():
            test_for_length = 3  # 3 columns 
            if forcealigned: 
                test_for_length = 5
            if len(t) == test_for_length:
                tmp = t[-1]
                if tmp == "!ENTER" or tmp == '!EXIT':
                    tmp = 'sil'
                if tmp in foldings:
                    tmp = foldings[tmp]
                res.append(tmp)
                if t[-1] == "!EXIT":
                    res.append('\n')
    return res


if __name__ == '__main__':
    arguments = docopt(__doc__, version='mlf_to_text version 0.1')
    foldings = {}
    if arguments['--timitfoldings']:
        with open('timit_foldings.json') as f:
            foldings = json.load(f)
        print >> sys.stderr, "using foldings", foldings
    text_list = parse(sys.stdin, arguments['--forcealigned'], foldings)
    print >> sys.stderr, "number of phones", len(set(text_list))-1  # -1 for\n
    print " ".join(text_list)
