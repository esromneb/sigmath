import os
import collections
import csv
import difflib
import errno
import hashlib
import itertools
import logging
import math
import numpy as np
import random
import scipy
import socket
import string
import struct
import sys
import time
from itertools import chain, repeat
from numpy.fft import fft, fftshift
# from scipy import ndimage
import numpy as np
import inspect
import matplotlib.pyplot as plt
# import pyximport;pyximport.install()
import zmq

from osibase import OsiBase
from filterelement import FilterElement
# from sigmathcython import *

# converts string types to complex
def raw_to_complex(str):
    f1 = struct.unpack('%df' % 1, str[0:4])
    f2 = struct.unpack('%df' % 1, str[4:8])

    f1 = f1[0]
    f2 = f2[0]
    return f1 + f2*1j

def complex_to_raw(n):

    s1 = struct.pack('%df' % 1, np.real(n))
    s2 = struct.pack('%df' % 1, np.imag(n))

    return s1 + s2

# converts complex number to a pair of int16's, called ishort in gnuradio
def complex_to_ishort(c):
    short = 2**15-1
    re = struct.pack("h", np.real(c)*short)
    im = struct.pack("h", np.imag(c)*short)
    return re+im

# i first, q second
def complex_to_ishort_multi(floats, endian=""):
    rr = np.real(floats)
    ii = np.imag(floats)
    zz = np.array((rr,ii)).transpose()
    zzz = zz.reshape(len(floats)*2) * (2**15-1)
    bytes = struct.pack(endian+"%sh" % len(zzz), *zzz)
    return bytes

# q first, i second
def complex_to_ishort_multi_rev(floats, endian=""):
    rr = np.real(floats)
    ii = np.imag(floats)
    zz = np.array((ii,rr)).transpose()
    zzz = zz.reshape(len(floats)*2) * (2**15-1)
    bytes = struct.pack(endian+"%sh" % len(zzz), *zzz)
    return bytes

# i first, q second
def ishort_to_complex_multi(ishort_bytes, endian=""):
    packed = struct.unpack(endian+"%dh" % int(len(ishort_bytes)/2), ishort_bytes)
    rere = sig_everyn(packed, 2, 0)
    imim = sig_everyn(packed, 2, 1)
    floats_recovered = list(itertools.imap(np.complex, rere, imim))
    return floats_recovered

# q first, i second
def ishort_to_complex_multi_rev(ishort_bytes, endian=""):
    packed = struct.unpack(endian+"%dh" % int(len(ishort_bytes)/2), ishort_bytes)
    rere = sig_everyn(packed, 2, 1)
    imim = sig_everyn(packed, 2, 0)
    floats_recovered = list(itertools.imap(np.complex, rere, imim))
    return floats_recovered

def complex_to_raw_multi(floats, endian=""):
    rr = np.real(floats)
    ii = np.imag(floats)
    zz = np.array((rr,ii)).transpose()
    zzz = zz.reshape(len(floats)*2)
    bytes = struct.pack(endian+"%sf" % len(zzz), *zzz)
    return bytes

def raw_to_complex_multi(raw_bytes):
    packed = struct.unpack("%df" % int(len(raw_bytes)/4), raw_bytes)
    rere = sig_everyn(packed, 2, 0)
    imim = sig_everyn(packed, 2, 1)
    floats_recovered = list(itertools.imap(np.complex, rere, imim))
    return floats_recovered

# a pretty-print for hex strings
def get_rose(data):
    try:
        ret = ' '.join("{:02x}".format(ord(c)) for c in data)
    except TypeError, e:
        ret = str(data)
    return ret

def print_rose(data):
    print get_rose(data)

def get_rose_int(data):
    # adding int values
    try:
        ret = ' '.join("{:02}".format(ord(c)) for c in data)
    except TypeError, e:
        ret = str(data)
    return ret

# if you want to go from the pretty print version back to a string (this would not be used in production)
def reverse_rose(input):
    orig2 = ''.join(input.split(' '))
    orig = str(bytearray.fromhex(orig2))
    return orig

# this is meant to replace print comma like functionality for ben being lazy
def s_(*args):
    out = ''
    for arg in args:
        out += str(arg)+' '
    out = out[:-1]
    return out

def print_hex(str, ascii = False):
    print 'hex:'
    tag = ''
    for b in str:
        if ascii:
            if b in string.printable:
                tag = b
            else:
                tag = '?'
        print ' ', format(ord(b), '02x'), tag

def print_dec(str):
    print 'hex:'
    for b in str:
        print ' ', ord(b)

# http://stackoverflow.com/questions/10237926/convert-string-to-list-of-bits-and-viceversa
def str_to_bits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def bits_to_str(bits):
    chars = []
    for b in range(len(bits) / 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def all_to_ascii(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(all_to_ascii, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(all_to_ascii, data))
    else:
        return data

def floats_to_bytes(rf):
    return ''.join([complex_to_ishort(x) for x in rf])

def bytes_to_floats(rxbytes):
    packed = struct.unpack("%dh" % int(len(rxbytes)/2), rxbytes)
    rere = sig_everyn(packed, 2, 0)
    imim = sig_everyn(packed, 2, 1)
    assert len(rere) == len(imim)
    rx = list(itertools.imap(np.complex, rere, imim))

def drange(start, end=None, inc=None):
    """A range function, that does accept float increments..."""
    import math

    if end == None:
        end = start + 0.0
        start = 0.0
    else: start += 0.0 # force it to be a float

    if inc == None:
        inc = 1.0
    count = int(math.ceil((end - start) / inc))

    L = [None,] * count

    L[0] = start
    for i in xrange(1,count):
        L[i] = L[i-1] + inc
    return L




def unroll_angle(input):
    thresh = np.pi

    adjust = 0

    sz = len(input)

    output = [None]*sz

    output[0] = input[0]

    for index in range(1,sz):
        samp = input[index]
        prev = input[index-1]

        if(abs(samp-prev) > thresh):
            direction = 1
            if( samp > prev ):
                direction = -1
            adjust = adjust + 2*np.pi*direction

        output[index] = input[index] + adjust

    return output

def bits_cpm_range(bits):
    bits = [(b*2)-1 for b in bits] # convert to -1,1
    return bits

def bits_binary_range(bits):
    bits = [int((b+1)/2) for b in bits]  # convert to ints with range of 0,1
    return bits

def ip_to_str(address):
    return socket.inet_ntop(socket.AF_INET, address)


def nonblock_read(sock, size):
    try:
        buf = sock.read()
    except IOError, e:
        err = e.args[0]
        if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
            return None  # No data available
        else:
            # a "real" error occurred
            print e
            sys.exit(1)
    else:
        if len(buf) == 0:
            return None
        return buf

# returns None if socket doesn't have any data, otherwise returns a list of bytes
# you need to set os.O_NONBLOCK on the socket at creation in order for this function to work
#   fcntl.fcntl(sock, fcntl.F_SETFL, os.O_NONBLOCK)
def nonblock_socket(sock, size):
    # this try block is the non blocking way to grab UDP bytes
    try:
        buf = sock.recv(size)
    except socket.error, e:
        err = e.args[0]
        if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
            return None  # No data available
        else:
            # a "real" error occurred
            print e
            sys.exit(1)
    else:
        # got data
        return buf

def nonblock_socket_from(sock, size):
    # this try block is the non blocking way to grab UDP bytes
    try:
        buf, addr = sock.recvfrom(size)
    except socket.error, e:
        err = e.args[0]
        if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
            return None, None  # No data available
        else:
            # a "real" error occurred
            print e
            sys.exit(1)
    else:
        # got data
        return buf, addr


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def save_rf_grc(filename, data):
    dumpfile = open(filename, 'w')
    for s in data:
        dumpfile.write(complex_to_raw(s))
    dumpfile.close()

def read_rf_grc(filename, max_samples=None):
    file = open(filename, 'r')
    piece_size = 8
    dout = []
    sample_count = 0
    while True:
        bytes = file.read(piece_size)

        if bytes == "":
            break  # end of file
        dout.append(raw_to_complex(bytes))
        sample_count += 1
        if max_samples is not None and sample_count >= max_samples:
            break

    return dout



# write rf to a csv file
# to read file from matlab
# k = csvread('Y:\home\ubuntu\python-osi\qam4.csv');
# kc = k(:,1) + k(:,2)*1j;
def save_rf(filename, data):
    dumpfile = open(filename, 'w')
    for s in data:
        print >> dumpfile, np.real(s), ',', np.imag(s)
    dumpfile.close()

# read rf from a csv file
# if your file is dc in matlab, run this:
#   dcs = [real(dc) imag(dc)];
#   csvwrite('filename.csv',dcs);
#   csvwrite('h3packetshift.csv', [real(dc) imag(dc)]);
def read_rf(filename):
    # read a CSV file
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=',', quotechar='|')
    data = []
    for row in reader:
        data.append(float(row[0]) + float(row[1])*1j)
    file.close()
    return data

def read_rf_hack(filename):
    # read a CSV file
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=',', quotechar='|')
    data = []
    for row in reader:
        data.append(float(row[0]))
        data.append(float(row[1]))
    file.close()
    return data

# basic logging setup
def setup_logger(that, name, prefix=None):
    if prefix is None:
        prefix = name

    that.log = logging.getLogger(name)
    that.log.setLevel(logging.INFO)
    # create console handler and set level to debug
    lch = logging.StreamHandler()
    lch.setLevel(logging.INFO)
    lfmt = logging.Formatter(prefix+': %(message)s')
    # add formatter to channel
    lch.setFormatter(lfmt)
    # add ch to logger
    that.log.addHandler(lch)

# convert a list of bits to an unsigned int
# h is the number of bits we are expecting in the list
def bit_list_unsigned_int(lin, h):
    sym = 0
    for j in range(h):
        try:
            sym += lin[j]*2**(h-j-1)
        except IndexError:
            sym += 0
    return sym

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return (idx,array[idx])




def tone_gen(samples, fs, hz):
    assert type(samples) == type(0)
    assert type(fs) in (int, long)

    inc = 1.0/fs * 2 * np.pi * hz

    if hz == 0:
        args = np.array([0] * samples)
    else:
        args = np.linspace(0, (samples-1) * inc, samples) * np.array([1j] * samples)
    return np.exp(args)

## pass in raw rf data first, and your ideal search wave second
# pass show to nplot the result
# pass everything to return all data, defaults to false
def typical_xcorr(data, search, show=False,everything=False):
    xcr = np.correlate(data, search, 'full')
    absxcr = abs(xcr)  # save absolute value
    xcrscore, xcridx = sig_max(absxcr) # find the max of the abs (score), and the index of that
    peakxcr = xcr[xcridx]  # apply the index to the non abs xcr, to get a sample we can take the angle of
    sampleangle = np.angle(peakxcr)

    if show:
        nplot(absxcr, "abs of xcorr output")

    if everything:
        return xcridx, peakxcr, sampleangle, xcrscore, xcr
    else:
        return xcridx, peakxcr, sampleangle
    #fixme, get lags out


def mat_nicename(H):
    return "Matrix (h,w) %d %d"%(H.shape[0],H.shape[1])

# inplace modifies H
def mat_switch_rows(H, ida, idb):
    t = np.copy(H[ida,:])
    H[ida,:] = H[idb,:]
    H[idb,:] = t
    return None

# inplace
def mat_switch_cols(H, ida, idb):
    t = np.copy(H[:,ida])
    H[:,ida] = H[:,idb]
    H[:,idb] = t
    return None



def o_map_psk(din):

    res = []

    pairwise = zip(din[::2], din[1::2])

    for cut in pairwise:
        if cut == (0,0):
            res.append(2)
        elif cut == (0,1):
            res.append(1)
        elif cut == (1,0):
            res.append(3)
        elif cut == (1,1):
            res.append(0)

    return res



# Returns baked output from the matlab command:
#  rrcFilter = rcosdesign(rolloff, span, sps);
# if a new set of params is passed, this returns None
def o_rrc_filter_lookup(rolloff, span, sps):
    rrcFilter = None

    if rolloff == 0.25 and span == 6 and sps == 4:
        rrcFilter = [-0.0187733440045221, 0.003013558666422, 0.0326772345462546, 0.0470935833402524, 0.0265495177022908, -0.0275222240300177, -0.0852248750170832, -0.0994474359919262, -0.0321472673983147, 0.119037148179254, 0.311176411555479, 0.472003895565433, 0.534632070178156, 0.472003895565433, 0.311176411555479, 0.119037148179254, -0.0321472673983147, -0.0994474359919262, -0.0852248750170832, -0.0275222240300177, 0.0265495177022908, 0.0470935833402524, 0.0326772345462546, 0.003013558666422, -0.0187733440045221]
    elif rolloff == 0.25 and span == 8 and sps == 6:
        rrcFilter = [0.00866532792410113, 0.00658252680452728, 0.000689062625426406, -0.0074712343335428, -0.0149894394106519, -0.0184751723085405, -0.0153182803408426, -0.00491214847998965, 0.0106347206341242, 0.0266632859560034, 0.0371116854587008, 0.0364198115321213, 0.0216633198102528, -0.00577076737254631, -0.0396028976063981, -0.069540009878394, -0.083558519268118, -0.0710847620356398, -0.0262308544541024, 0.0498600699425154, 0.148327139837451, 0.253907215811741, 0.348041615194585, 0.413040313779774, 0.436237887518672, 0.413040313779774, 0.348041615194585, 0.253907215811741, 0.148327139837451, 0.0498600699425154, -0.0262308544541024, -0.0710847620356398, -0.083558519268118, -0.069540009878394, -0.0396028976063981, -0.00577076737254631, 0.0216633198102528, 0.0364198115321213, 0.0371116854587008, 0.0266632859560034, 0.0106347206341242, -0.00491214847998965, -0.0153182803408426, -0.0184751723085405, -0.0149894394106519, -0.0074712343335428, 0.000689062625426406, 0.00658252680452728, 0.00866532792410113]
    elif rolloff == 0.25 and span == 10 and sps == 8:
        rrcFilter =  [-0.00265291895695268, -0.0038367285758387, -0.00404178046649392, -0.00307918477864112, -0.00103711397846142, 0.00170441339943213, 0.0045271711482077, 0.00669247991307441, 0.00750358793759834, 0.00648076379403586, 0.00351054324124175, -0.00106825247595824, -0.00646958364590197, -0.0115675402132076, -0.0150962227680158, -0.0159112035315436, -0.0132645947847634, -0.0070368939631878, 0.0021292761993053, 0.0128349930785245, 0.0230886023735744, 0.0306157954214996, 0.0332746952178824, 0.0295092014286344, 0.0187589698439958, 0.00174365601293929, -0.0194462504520092, -0.0414910755739182, -0.0602169454952409, -0.0711634708322952, -0.0702661145770751, -0.0545489915088333, -0.0227141459386971, 0.0244771507857901, 0.0841075268503736, 0.151324993567835, 0.219866476897559, 0.282817489474821, 0.333501607917967, 0.366369319357036, 0.37775250739262, 0.366369319357036, 0.333501607917967, 0.282817489474821, 0.219866476897559, 0.151324993567835, 0.0841075268503736, 0.0244771507857901, -0.0227141459386971, -0.0545489915088333, -0.0702661145770751, -0.0711634708322952, -0.0602169454952409, -0.0414910755739182, -0.0194462504520092, 0.00174365601293929, 0.0187589698439958, 0.0295092014286344, 0.0332746952178824, 0.0306157954214996, 0.0230886023735744, 0.0128349930785245, 0.0021292761993053, -0.0070368939631878, -0.0132645947847634, -0.0159112035315436, -0.0150962227680158, -0.0115675402132076, -0.00646958364590197, -0.00106825247595824, 0.00351054324124175, 0.00648076379403586, 0.00750358793759834, 0.00669247991307441, 0.0045271711482077, 0.00170441339943213, -0.00103711397846142, -0.00307918477864112, -0.00404178046649392, -0.0038367285758387, -0.00265291895695268]
    elif rolloff == 0.5 and span == 10 and sps == 8:
        rrcFilter = [-0.000227360782730039200000,0.001088738429480835300000,0.002084746204746209700000,0.002371375200663914500000,0.001768451863687256400000,0.000392002682105138340000,-0.001351170708598336100000,-0.002870096591219498600000,-0.003572812300043569700000,-0.003072796427970129000000,-0.001351170708598331300000,0.001186149711456712000000,0.003789539707901269900000,0.005576181317467936900000,0.005816879270585570100000,0.004213430128940225500000,0.001071843690013067400000,-0.002702793301169967200000,-0.005816879270585570900000,-0.006968996504726427200000,-0.005305355591061771000000,-0.000815566309429241660000,0.005469024296707546000000,0.011574239347020513000000,0.015005811660182994000000,0.013431057725172559000000,0.005469024296707539000000,-0.008615925345214449000000,-0.026526777955308903000000,-0.044010343226686024000000,-0.055454249046249099000000,-0.054933726077336487000000,-0.037514529150457464000000,-0.000532406049215232180000,0.055454249046249134000000,0.126292748642899940000000,0.204584829987635860000000,0.280739008412446870000000,0.344548530692574790000000,0.386989294703540040000000,0.401870228675680950000000,0.386989294703540040000000,0.344548530692574790000000,0.280739008412446870000000,0.204584829987635860000000,0.126292748642899940000000,0.055454249046249134000000,-0.000532406049215232180000,-0.037514529150457464000000,-0.054933726077336487000000,-0.055454249046249099000000,-0.044010343226686024000000,-0.026526777955308903000000,-0.008615925345214449000000,0.005469024296707539000000,0.013431057725172559000000,0.015005811660182994000000,0.011574239347020513000000,0.005469024296707546000000,-0.000815566309429241660000,-0.005305355591061771000000,-0.006968996504726427200000,-0.005816879270585570900000,-0.002702793301169967200000,0.001071843690013067400000,0.004213430128940225500000,0.005816879270585570100000,0.005576181317467936900000,0.003789539707901269900000,0.001186149711456712000000,-0.001351170708598331300000,-0.003072796427970129000000,-0.003572812300043569700000,-0.002870096591219498600000,-0.001351170708598336100000,0.000392002682105138340000,0.001768451863687256400000,0.002371375200663914500000,0.002084746204746209700000,0.001088738429480835300000,-0.000227360782730039200000]

    return rrcFilter


# pass in any sparse, this will convert to bsr sparse and save
def fancy_save_sparse(filename, H):
    Hsparse = scipy.sparse.bsr_matrix(H, dtype=np.int16)
    save_sparse_csr(filename, Hsparse)

# def fancy_load_sparse(filename):
#     loader = np.load(filename)
#     bsr = scipy.sparse.bsr_matrix((  loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])


# only pass type of bsr
def save_sparse_csr(filename, array):
    assert type(array) == scipy.sparse.bsr.bsr_matrix
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.bsr_matrix((  loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])














def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N



# ---------------- Beginning of matplotlib
##
# @defgroup nplot Plotting related helper functions
# @{
#

_nplot_figure = 0

##
#  Plot the sparsity pattern of 2-D array
# @param A the 2-D array
# @param title The plot title (string)
def nplotspy(A, title=""):
    fig = nplotfigure()

    plt.title(title)
    plt.spy(A, precision=0.01, markersize=3)

    return fig

def ncplot(rf, title=None, newfig=True):
    fig = None


    if newfig:
        fig = nplotfigure()

    if title is not None:
        plt.title(title)

    plt.plot(np.real(rf))
    plt.plot(np.imag(rf), 'r')
    return fig

def nplot(rf, title=None, newfig=True):
    fig = None


    if newfig:
        fig = nplotfigure()

    if title is not None:
        plt.title(title)

    plt.plot(np.real(rf))
    return fig

def nplotdots(rf, title="", hold=False):
    fig = None
    if hold:
        plt.hold(True)
    else:
        fig = nplotfigure()
        plt.title(title)
    plt.plot(range(len(rf)), np.real(rf), '-ko')
    return fig


def nplotqam(rf, title="", newfig=True):
    fig = None

    if newfig:
        fig = nplotfigure()

    if title is not None:
        plt.title(title)

    plt.plot(np.real(rf), np.imag(rf), '.b', alpha=0.6)
    return fig


def nplotfftold(rf, title="", hold=False):
    fig = None

    if hold:
        plt.hold(True)
    else:
        fig = nplotfigure()
        plt.title(title)

    plt.plot(abs(fftshift(fft(rf))))
    return fig

def sig_peaks(bins, hz, num = 1, peaksHzSeparation = 1):
    binsabs = np.abs(bins)

    llen = len(bins)

    # approx method, assumes all bins are evenly spaces (they better be)
    binstep = hz[1] - hz[0]
    hzsepinbins = np.ceil(peaksHzSeparation / binstep)

    res = []

    for i in range(num):
        maxval,peakidx = sig_max(binsabs)
        res.append(peakidx)

        lowbin = int(np.max([0,peakidx-hzsepinbins]))
        highbin = int(np.min([llen,peakidx+hzsepinbins]))

        binsabs[lowbin:highbin] = 0.


    return res


def nplotfft(rf, fs = 1, title=None, newfig=True, peaks=False, peaksHzSeparation=1, peaksFloat=1.1):
    fig = None

    if newfig is True:
        fig = nplotfigure()

    if title is not None:
        plt.title(title)

    N = len(rf)

    X = fftshift(fft(rf)/N)
    absbins = 2*np.abs(X)
    df = fs/N
    f = np.linspace(-fs/2, fs/2-df, N)

    plt.plot(f, absbins)
    plt.xlabel('Frequency (in hertz)')
    plt.ylabel('Magnitude Response')


    if peaks:
        peaksres = sig_peaks(X, f, peaks, peaksHzSeparation)
        if type(newfig) is type(True):
            ax = fig.add_subplot(111)
        else:
            ax = newfig


        for pk in peaksres:

            maxidx = pk
            maxval = absbins[pk]

            lbl = s_('hz:', f[maxidx])

            ax.annotate(lbl, xy=(f[maxidx], maxval), xytext=(f[maxidx], maxval * peaksFloat),
                        arrowprops=dict(facecolor='black'),
                        )

    return fig


def nplothist(x, title = "", bins = False):
    fig = nplotfigure()

    plt.title(title)

    # more options
    # n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

    if bins is not False:
        plt.hist(x, bins)
    else:
        plt.hist(x)

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    return fig


def nplotber(bers, ebn0, titles, title = ""):
    assert(len(bers) == len(ebn0) == len(titles))

    fig = nplotfigure()

    maintitle = "BER of ("

    for i in range(len(bers)):
        # check if any values are positive
        if any(k > 0 for k in bers[i]):
            # if so plot normally
            plt.semilogy(ebn0[i], bers[i], '-s', linewidth=2)
            maintitle += titles[i] + ', '
        else:
            # if all values are zero, don't plot (doing so forever prevents this plot window from drawing new lines)
            maintitle += 'ValueError: Data has no positive values, and therefore can not be log-scaled.' + ', '


    maintitle += ")"


    gridcolor = '#B0B0B0'
    plt.grid(b=True, which='major', color=gridcolor, linestyle='-')
    plt.grid(b=True, which='minor', color=gridcolor, linestyle='dotted')

    plt.legend(titles)
    plt.xlabel('Eb/No (dB)')
    plt.ylabel('BER')

    if title == "":
        plt.title(maintitle)
    else:
        plt.title(title)

    return fig


def nplotmulti(xVals, yVals, legends, xLabel ='x', yLabel ='y', title ='', semilog = False, newfig=True, style='-s'):
    assert(len(yVals) == len(xVals) == len(legends))

    fig = None
    if newfig:
        fig = nplotfigure()
        plt.title(title)

    for i in range(len(yVals)):
        if semilog == True:
            plt.semilogy(xVals[i], yVals[i], style, linewidth=2)
        elif semilog == False:
            plt.plot(xVals[i], yVals[i], style, linewidth=2)

    gridcolor = '#B0B0B0'
    plt.grid(b=True, which='major', color=gridcolor, linestyle='-')
    plt.grid(b=True, which='minor', color=gridcolor, linestyle='dotted')

    plt.legend(legends)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)



    return fig


def nplotangle(angle, title = None, newfig=True):
    global _nplot_figure
    fig = None
    if newfig:
        fig = plt.figure(_nplot_figure, figsize=(7.,6.))
        _nplot_figure += 1

    res = 100
    cir = np.exp(1j*np.array(range(res+1))/(res/np.pi/2))
    cir = np.concatenate(([0],cir))

    cir = cir * np.exp(1j*angle)

    plt.plot(np.real(cir), np.imag(cir), 'b', alpha=0.6, linewidth=3.0)
    if title is not None:
        plt.title(title)
    else:
        plt.title(s_("angle:", angle))
    return fig


def nplotshow():
    plt.show()


def nplotfigure():
    global _nplot_figure

    fig = plt.figure(_nplot_figure)
    _nplot_figure += 1

    return fig


def nplotresponselinear(b, a, cutoff, fs):
    w, h = scipy.signal.freqz(b, a, worN=8000)
    nplotfigure()
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Linear Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()



## Plots frequency response of digital filter
# @param b b coefficients
# @param a a coefficients
# @param mode set mode 'frequency' and also pass in fs to display units in hz, or set mode 'radians'
# @param title graph title
# @param fs fs of filter in 'frequency' mode
# @param cutoff draw a vertial line at the filters cutoff if you know it
def nplotresponse(b, a, mode='frequency', title='', fs=0, cutoff=None):
    w, h = scipy.signal.freqz(b, a)

    fig = nplotfigure()
    plt.title(title)
    ax1 = fig.add_subplot(111)

    if mode == 'frequency':
        wplot = 0.5 * fs * w / np.pi
        xlab = 'Frequency [Hz]'
    elif mode == 'radians':
        wplot = w
        xlab = 'Frequency [rad/sample]'
    else:
        assert 0

    plt.plot(wplot, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel(xlab)
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(wplot, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')

    if cutoff is not None:
        plt.axvline(cutoff, color='k')

    return fig


def nplottext(str, newfig=True):
    fig = None
    if newfig == True:
        fig = nplotfigure()
        plt.title('txt')
        ax = fig.add_subplot(111)
    else:
        ax = newfig

    ax.set_aspect(1)

    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax.text(0, 0, str, ha="center", va="center", size=14,
            bbox=bbox_props)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    return fig


##
# @}
#



def rand_string_ascii(len):
    return ''.join(random.choice(string.ascii_letters) for _ in range(len))



def sigflatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def sigdup(listt, count=2):
    # return list(flatten([y for y in repeat([y for y in listt], count)]))
    return list(sigflatten([repeat(x, count) for x in listt]))



## wrapper around numpy's built in FFT that returns frequency of each bin
# note that this function returns data that has already been fftshift'd
# @param data data to fft
# @param fs samples per second to be used for bin calculation
# @returns (fft_data, bins)
def sig_fft(data, fs):

    dt = 1 / fs     #                % seconds per sample
    N = len(data)
     # Fourier Transform:
    X = fftshift(fft(data)/N)
    # %% Frequency specifications:
    dF = float(fs) / N     #                % hertz
    # f = -Fs/2:dF:Fs/2-dF;#           % hertz
    bins = drange(-fs / 2, fs / 2, dF)

    return X, bins
    # %% Plot the spectrum:
    # figure;
    # plot(f,2*abs(X));
    # xlabel('Frequency (in hertz)');
    # title('Magnitude Response');
    # disp('Each bin is ');
    # disp(dF);


def sig_rms(x):
    return np.sqrt(np.mean(x * np.conj(x)))



# returns the FIRST occurrence of the maximum
# this is the fastest by FAR, but it requires that data be an ndarray
# behaves like matlab max() do not pass in complex values and assume that this works
def sig_max(data):
    idx = np.argmax(data)
    m = data[idx]
    return (m, idx)

def sig_max2(data):
    m = max(data)
    idx = [i for i, j in enumerate(data) if j == m]
    idx = idx[0]
    return (m, idx)

def sig_max3(data):
    m = max(data)
    idx = data.index(m)
    return (m, idx)

def sig_max4(data):
    m = max(data)
    idx = data.index(m)
    return (m, idx)


def sig_everyn(data, n, phase=0):
    l = len(data)
    assert phase < n, "Phase must be less than n"
    assert n <= l, "n must be less than or equal length of data"

    # subtract the phase from length, and then grab that many elements from the end of data
    # this is the same as removing 'phase' elements from the beginning
    if phase != 0:
        phaseapplied = data[-1*(l-phase):]

        # after applying phase, return every nth element from that
        return phaseapplied[::n]
    else:
        return data[::n]

# http://stackoverflow.com/questions/17904097/python-difference-between-two-strings/17904977#17904977

def sig_diff(a,b,max=False):
    at = time.time()
    count = 0
    # print('{} => {}'.format(a,b))
    for i,s in enumerate(difflib.ndiff(a, b)):
        if s[0]==' ': continue
        elif s[0]=='-':
            try:
                c = u'{}'.format(s[-1])
            except:
                c = ' '
            print(u'Delete "{}" from position {}'.format(c,i))
        elif s[0]=='+':
            print(u'Add "{}" to position {}'.format(s[-1],i))
        count += 1
        if max and count >= max:
            print "stopping after", count, "differences"
            break
    print()
    bt = time.time()
    print "sig_diff() ran in ", bt-at

def sig_diff2(a, b, unused=False):
    lmin = min(len(a),len(b))
    lmax = max(len(a),len(b))

    errors = 0
    for i in range(lmin):
        if a[i] != b[i]:
            errors += 1

    errors += lmax-lmin
    if lmax != lmin:
        print "Strings were", errors, "different with additional", lmax-lmin, "due to length differences"
    else:
        print "Strings were", errors, "different"


# builds a ring of evenly spaced points, used for circular QAM
def buildQAMRing(r, count, rot=0):
        ainc = 1.0/count * 2 * np.pi

        points = [0] * count

        for i in range(count):
            p = np.e ** (1j * (rot + ainc*i))
            p *= r  # scale to amplitude for the ring
            points[i] = p
        return points

# returns the closest distance between a list of complex points
# used for checking the quality of QAM constellations
def listClosestDistance(points, verbose=True):

        best = 999999
        closestpair = (-1,-1)

        for i in range(len(points)):
            pouter = points[i]
            for k in range(len(points)):
                if i == k:
                    continue
                pinner = points[k]
                diff = min(best, abs(pouter-pinner))
                if diff < best:
                    best = diff
                    closestpair = (pouter,pinner)
        if verbose:
            print "closest", best, closestpair

        return (best,closestpair)

def sig_awgn(data, snrdb):
    ll = len(data)

    # only way I know to make complex noise
    noise = np.random.normal(0, 1, ll) + 0j
    rot = np.random.random_sample(ll)
    for i in range(len(noise)):
        noise[i] = noise[i] * np.exp(1j*rot[i]*2.0*np.pi) # random rotation with same amplitude

    sigma2 = 10.0**(-snrdb/10.0)
    out = data + noise*math.sqrt(sigma2)

    return out


## When using lower triangular matrices, this function will return the correct index for the relationship between any to indices
# (aka mirrors an upper triangular index to lower triangular index)
# @param a An index
# @param b An index
# @returns an index that is always lower triangular, as a tuple
def lower_triangular(a, b):
    if b > a:
        idx = (b, a)
    else:
        idx = (a, b)
    return idx

## Given N objects, returns unique pairs of objects
# Useful for upper/lower triangular matrices
# @param dim Number of objects
# @returns set() of tuples of unique objects
def unique_matrix_pair(dim, lt=False):
    s = set()
    for i in range(dim):
        for j in range(dim):
            if i != j:
                if (j, i) not in s:
                    if lt:
                        s.add(lower_triangular(i,j))
                    else:
                        s.add((i, j))
    return s


# accepts numpy matrices, always uses int16 math to compute sha
def sig_sha256_matrix(H):
    marshal = np.int16(H).tolist()
    return sig_sha256(marshal)

# converts the sparse matrix into dense, then returns the sha256
def sig_sha256_sparse_matrix(H):
    return sig_sha256_matrix(H.todense())

def sig_sha256(as_str):
    worker = hashlib.sha256()
    worker.update(str(as_str))
    return worker.digest()

def sig_lin_interp(a,b,ratio):
    # print "a", a
    # print "b", b
    # print "r", ratio
    slope = b-a
    return a+(slope*ratio)

def linear_unwrap(vec):
    llen = len(vec)
    std = -1
    idx = -1
    for i in range(llen):
        rolled = np.roll(vec, i)
        unwrap = np.unwrap(rolled)

        sstd = np.std(np.diff(unwrap))

        if std == -1:
            std = sstd
            idx = i
        elif sstd < std:
            std = sstd
            idx = i

    return (np.unwrap(np.roll(vec, idx)), idx)

def circle_shift_line(vec):
    llen = len(vec)
    mmin = -1
    minindex = -1
    for i in range(llen):
        vec_shift = np.roll(vec,i)
        s = np.std(np.diff(vec_shift))
        if minindex == -1:
            mmin = s
            minindex = i
        elif s < mmin:
            mmin = s
            minindex = i

    rolled = np.roll(vec, minindex)
    return (rolled,minindex)

def linear_zero_cross(vec):
    lhs = -1
    for i in range(len(vec)):
        sam = vec[i]
        if lhs == -1:
            if sam < 0:
                lhs = i - 1
                break
    rhs = lhs + 1
    lhss = vec[lhs]
    rhss = vec[rhs]
    slope = rhss - lhss

    zerocross = -lhss / slope + lhs
    return zerocross
        # print "lhs was", lhs
        # print "slope", slope
        # print "zerocross", zerocross

# circle shifts so that the peek is in the center, and the slope of the sides are always continuous
def circle_shift_peak(score_vec):
    llen = 8
    desired_centers = [3,4]
    assert len(score_vec) == llen, "circle_shift_peak only works for length 8 vectors"


    [maxval, maxidx] = sig_max(score_vec)
    beststd = 0
    bestidx = -1
    for desired in desired_centers:
        adj = desired - maxidx
        rolled = np.roll(score_vec, adj)
        title = s_('dataset', repr(score_vec)[0:5], ']  centered at', str(desired))

        # the set with the smaller std deviation of the double derivative is the one we want
        stddev = np.std(np.diff(np.diff(rolled)))

        if bestidx == -1:
            beststd = stddev
            bestidx = desired
        elif stddev < beststd:
            beststd = stddev
            bestidx = desired

    # txt = s_('dataset', repr(score_vec)[0:5], ']  best is ', bestidx)
    # print txt

    shift = bestidx-maxidx
    rolled = np.roll(score_vec, shift)

    # return a tuple with the shifted array and how much we did it by
    return (rolled, shift)


def sig_ms():
    import time
    millis = int(round(time.time() * 1000))
    return millis

## Pass samples of a peak
# Will polyfit, and then return maximum of fit curve
# @param peak's data samples, ususally I pass 8
# @param order order of polyfit, i use 4
# @param doplot pass True for very nice plot of what's going on
def polyfit_peak_climb(data, order, doplot = False):
    # http://stackoverflow.com/questions/29634217/get-minimum-points-of-numpy-poly1d-curve
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html

    # assume that interp os over 0-len(data)
    llen = len(data)
    x = range(llen)
    p = np.polyfit(x, data, order)
    pp = np.poly1d(p)  # create an object for easy lookups

    if doplot:
        # dont need to do costly linspace
        xp = np.linspace(x[0], x[-1], 100)
        nplotfigure()
        plt.plot(x, data, '.', xp, pp(xp))



    # this takes the deriv and idk exactly how it works
    crit = pp.deriv().r
    r_crit = crit[crit.imag == 0].real
    test = pp.deriv(2)(r_crit)
    x_max = r_crit[test < 0]

    if doplot:
        y_min = pp(x_max)
        plt.plot(x_max, y_min, 'o')
        xc = np.arange(0, 7, 0.02)
        yc = pp(xc)
        plt.plot(xc, yc)

    # with super high orders sometimes two maxima can be found
    # this attempts to remove any maxima that are outside the area of interest
    if len(x_max) > 1:
        # these two lines make bool arrays for the given conditions
        validleft = x_max >= 0
        validright = x_max <= llen
        valid = []
        for j in range(len(x_max)):
            if validleft[j] and validright[j]:
                valid.append(x_max[j])

        assert len(valid) == 1, "More than one maxima found in range of interest"
        x_max = valid


    # due to weirdness above, always unencapsulate off the only list element
    return x_max[0]



def fix_rad(rad):
    return math.fmod(rad, np.pi*2)
    # return rad % np.pi*2


def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def printOsiStack(stack):
    ptr = stack

    print "Bottom item is lowest:"

    lines = []

    while True:
        lines.append(s_('', ptr.name, type(ptr)))
        if not ptr.up:
            break
        ptr = ptr.up

    lines.reverse()

    for l in lines:
        print l


def osc_freq_drift(prev,time,Fc):
    f_delta_mean = time * -0.5e-9 / (2 * 3600) * Fc
    f_delta_variance = 0.2e-9 * Fc
    f_delta = f_delta_variance * (np.random.randn()) + f_delta_mean
    return prev+f_delta


def sigfdelayconvolution(signal, h):
    """ function that performs linear convolution """
    output = scipy.convolve(signal, h, "same")
    return output

## fractional delay
#
def sigfdelay(signal, N):
    assert N <= 1
    f = N
    i = 1

    # print "using f", f, "and i", i
    # print "using", [i - f, f]

    # perform linear interpolation for fractional delay
    output = sigfdelayconvolution(signal, [i - f, f])

    return output


def zero_insertion_upsample_complex(din, up):
    rem = up - 1
    llen = len(din)
    llenup = llen*up

    dout = [0.0j]*llenup
    dout[0::up] = din

    # This fn was built for upfirdn which does
    # zero inserion where the final zeros are excluded, ikd why but here goes
    return dout[:-rem]

def zero_insertion_upsample_real(din, up):
    # remainder, aka how many zeros between samples
    rem = up - 1

    # length of input
    llen = len(din)

    # length of output
    llenup = llen*up

    # magic indicing
    dout = [0.0]*llenup
    dout[0::up] = din

    # This fn was built for upfirdn which does
    # zero inserion where the final zeros are excluded, ikd why but here goes
    return dout[:-rem]

## dot.notation access to dictionary attributes
class ExposeDictionary(dict):
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

from stackcall import StackCall
from continuousiir import ContinuousIIR

def sig_upfirn_real(data,filter,up,down):

    dup = zero_insertion_upsample_real(data, up)
    print "len dup", len(dup)


    filtero = StackCall(ContinuousIIR(filter, [1.0]))

    print "lengh of input plus length of filter minus 1", len(dup) + len(filter)-1

    extra = max(0,len(filter)-1)

    option = np.concatenate((np.array(dup),np.array([0]*extra)))

    output = filtero.rxup(option)

    return output

def sig_upfirdn_up_complex(data, filter, up):

    dup = zero_insertion_upsample_complex(data, up)
    # print "len dup", len(dup)


    filtero = StackCall(ContinuousIIR(filter, [1.0]))

    # print "lengh of input plus length of filter minus 1", len(dup) + len(filter)-1

    extra = max(0,len(filter)-1)

    option = np.concatenate((np.array(dup),np.array([0.0j]*extra)))

    output = filtero.rxup(option)

    return output

def sig_upfirdn_down_complex(data,filter,span,down):

    filtero = StackCall(ContinuousIIR(filter, [1.0]))

    output = filtero.rxup(data)


    ds = output[0::down]

    return ds[span:]

def polyphase_sequence(period=8, periods=1, q=1, s=1):
    ''' Generate a polyphase sequence (q, s, and period should be relatively prime) '''
    x = np.zeros(period*periods, dtype=np.complex)
    if period & 1 == 0:
        for p in range(periods):
            x[period*p:period*(p+1)] = np.exp(-q*1j*np.pi*np.arange(period)**2.0/(s * period))
    else:
        for p in range(periods):
            x[period*p:period*(p+1)] = np.exp(-q*1j*np.pi*np.arange(period)*np.arange(-1, period-1)/(s * period))
    return x


def root_cosine(N, L, alpha, apply_window=False):
    ''' Generate root raised cosine pulse shape '''
    if (N & 1) == 0:
        # Even length
        t = np.arange(-int(N/2), int(N/2)) / float(L) + 0.5 / float(L)
    else:
        # Odd length
        t = np.arange(-int((N-1)/2), int((N-1)/2+1)) / float(L)

    h = np.zeros_like(t)

    for k in range(N):
        if t[k] == 0:
            h[k] = 1.0 - alpha + 4.0 * alpha / np.pi
        elif abs(t[k]) == 1.0 / (4.0 * alpha):
            a0 = (1.0 + 2.0 / np.pi) * np.sin(np.pi / 4.0 / alpha)
            a1 = (1.0 - 2.0 / np.pi) * np.cos(np.pi / 4.0 / alpha)
            h[k] = alpha / np.sqrt(2.0) * (a0 + a1)
        else:
            a0 = np.sin(np.pi * (1.0 - alpha) * t[k])
            a1 = 4.0 * alpha * t[k] * np.cos(np.pi * (1.0 + alpha) * t[k])
            a2 = np.pi * t[k] * (1.0 - (16.0 * alpha * alpha * t[k] * t[k]))
            h[k] = (a0 + a1) / a2

    # Hamming Window
    if apply_window:
        h *= 0.54 - 0.46 * np.cos(2*np.pi*np.arange(0, N) / (N - 1))

    return h



## Returns the current line number in our program.
def lineno(up = 0):
    if up == 0:
        return inspect.currentframe().f_back.f_lineno
    if up == 1:
        return inspect.currentframe().f_back.f_back.f_lineno

def sigtest_requires_gui(some_function):
    def wrapper(*args, **kwargs):
        if os.environ.has_key("UNITTEST_NO_X11"):
            return
        some_function(*args, **kwargs)
    return wrapper

# these are using the same variable for now, but it makes it
# eaiser because tests specifically say what they require
def sigtest_requires_octave(some_function):
    def wrapper(*args, **kwargs):
        if os.environ.has_key("UNITTEST_NO_X11"):
            return
        some_function(*args, **kwargs)
    return wrapper

def sigtest_artik_will_skip(some_function):
    def wrapper(*args, **kwargs):
        if os.environ.has_key("UNITTEST_NO_X11"):
            return
        some_function(*args, **kwargs)
    return wrapper



if os.environ.has_key("UNITTEST_NO_X11"):
    def nplot_dummy(*args, **kwargs):
        pass

    nplotspy = nplot_dummy
    nplot = nplot_dummy
    nplotdots = nplot_dummy
    nplotqam = nplot_dummy
    nplotfftold = nplot_dummy
    nplotfft = nplot_dummy
    nplothist = nplot_dummy
    nplotber = nplot_dummy
    nplotmulti = nplot_dummy
    nplotangle = nplot_dummy
    nplotshow = nplot_dummy
    nplotfigure = nplot_dummy
    nplotresponselinear = nplot_dummy
    nplotresponse = nplot_dummy
    nplottext = nplot_dummy

