import sys, os
sys.path.append(os.path.abspath('..'))
from sigmath import *


# print(str_to_bits('b'))


def test_bits():
    r = str_to_bits('b')
    # '0b1100010'
    e = [0,1,1,0,0,0,1,0]
    assert e == r




def test_csv():
    c0 = better_open_csv('data/a.csv')
    print(c0.header)
    assert c0.header[0] == 'head a'
    assert c0.data[2] == [4,4,3,4]

