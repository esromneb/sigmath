from random import *
import math
from sigmath import *
from numpy import real, imag, arange
from numpy.fft import fft, ifft, fftshift
from numpy.random import randint
from commpy.modulation import QAMModem
from osibase import OsiBase

# this import comes directly from QAMModem
from commpy.utilities import bitarray2dec, dec2bitarray

## Osi Layer wrapper around QamWrapper that automagically converts from bytes to -1,1 bits
class QAMLayer(OsiBase):
    ## Constructor
    def __init__(self, order):
        super(QAMLayer, self).__init__()           # this is the constructor for OsiBase
        self.mod = QAMWrapper(order)  ##!QAMWrapper instance

    ## Receive from up layer
    def rxdown(self, data, meta=False):
        bits = self.mod.demod(data)
        self.txup(bits_to_str(bits))

    ## Receive from down layer
    def rxup(self, data, meta=False):
        bits = str_to_bits(data)
        self.txdown(self.mod.modulate(bits))

    ## Unused
    def tick(self):
        pass

## Wrap comppy's Qam modulator for easier usage
class QAMWrapper(QAMModem):

    def bitarrayprecalc(self):
        self.table = {}  ##! Table to store the result

        ## Note that the earlier stages of the outer for loop are WASTEFUL mega
        # this double 4 loop ends up not only writing tuples like this (1,0,1,0) in the case of qam16
        # but also tuples with fewer bits.  These are used when the input needs to be padded at the end of the qam const
        for lookup_bits in range(1, self.num_bits_symbol+1):
            # jm is the number of shorter length tuples we are adding
            # the last loop of the outer forloop does what you would originally think this thing does
            for i in range(self.m):
                # note we grab only the ending N bits bits cuz ya
                bits = str_to_bits(chr(i))[-lookup_bits:]
                # print bits

                # note if we can call hash() on it, it can be indexed
                index = tuple(bits) # this is what we index on
                result = bitarray2dec(bits)

                # write to table
                self.table[index] = result



    # ==================== override ====================
    def modulate(self, input_bits):
        """ Modulate (map) an array of bits to constellation symbols.

        Parameters
        ----------
        input_bits : 1D ndarray of ints
            Inputs bits to be modulated (mapped).

        Returns
        -------
        baseband_symbols : 1D ndarray of complex floats
            Modulated complex symbols.

        """

        index_list = map(lambda i: self.table[tuple((input_bits[i:i+self.num_bits_symbol]))], \
                         xrange(0, len(input_bits), self.num_bits_symbol))
        baseband_symbols = self.constellation[index_list]

        return baseband_symbols

    # ==================== end override ====================

    ## In general, this is how we wrap "old style" classes that aren't inherited from Object
    # note the order of the line where we call __init__, compared to the more modern super() method
    # http://stackoverflow.com/questions/11527921/python-inheriting-from-old-style-classes
    def __init__(self, input1):
        assert input1 <= 256, "existing inherited modulator does not go over QAM256"
        QAMModem.__init__(self, input1)
        self.bitarrayprecalc()


    ## Modulate data
    # @returns modulated data divided by scalar
    def mod(self, data, **p):
        ret = self.modulate(data)
        scalar = self.lookup_scalar()
        return self.modulate(data) / scalar

    ## Demod with autogain.
    # This requires that packets be long enough to contain every point in the constellation
    def demod(self, data, autoscale=True):

        if len(data) == 0:
            return ""

        maxx = 1.0
        if autoscale:
            maxx = np.max(np.abs(data))
        scalar = self.lookup_scalar()
        return list(self.demodulate(np.array(data) / maxx * scalar, 'hard'))

    ## Soft demodulator
    def demod_soft(self, data, unknown_knob=1.5, autoscale=True):
        maxx = 1.0
        if autoscale:
            maxx = np.max(np.abs(data))
        scalar = self.lookup_scalar()
        return list(self.demodulate(np.array(data) / maxx * scalar, 'soft', unknown_knob))

    ## Lookup table for scalar
    def lookup_scalar(self):
        scale = 1.0
        if self.m == 256:
            scale = 21.2132034356
        elif self.m == 64:
            scale = 9.89949493661
        elif self.m == 32:
            scale = 7.08770047421
        elif self.m == 16:
            scale = 4.24264068712
        elif self.m == 8:
            scale = 2.83881568669
        elif self.m == 4:
            scale = 1.41421356237
        return scale

    ## Name, "QAM"
    def name(self):
        return "QAM"

    ## Homemade circular QAM
    def useCircularQAM256A(self):
        # this was built by hand, somehow the radius of the 3rd ring got too tight, messing up the even spacing
        ring1 = []
        ring1 += buildQAMRing(1.4142135623730951, 4)
        ring1 += buildQAMRing(3.41303,   10, 0.0314159265359)
        ring1 += buildQAMRing(5.401697,  17, 0.257610597594)
        ring1 += buildQAMRing(7.343697,  23, 0.521504380496)
        ring1 += buildQAMRing(9.343364,  29, 0.00837758040957)
        ring1 += buildQAMRing(11.343031, 35, 1.89333317256)
        ring1 += buildQAMRing(13.343031, 41, 1.26501464185)
        ring1 += buildQAMRing(15.34300,  48, 2.2787018714)
        ring1 += buildQAMRing(17.33963,  54, 0.0460766922527)  # this makes QAM261

        # because we have too many points, remove some
        for _ in range(5):
            (res,pair) = listClosestDistance(ring1, False)
            ring1.remove(pair[1])

        ring1 = np.array(ring1)
        maxx = abs(max(ring1))

        # until now this wrapper assumes that all qam constellations were scaled the same
        # now we need to scale our new one so it matches what the previous QAM256 looked like
        ring1 = ring1 * (self.lookup_scalar() / maxx)

        self.constellation = np.array(ring1)
