from osibase import OsiBase
from scipy.signal import butter, lfilter, freqz

## Filter data along one-dimension with an IIR or FIR filter.
# Guarantees same output regardless of chunking of input samples
class ContinuousIIR(OsiBase):
    ## Constructor
    # @param b B Coefficients
    # @param a A Coefficients
    def __init__(self, b, a):
        super(ContinuousIIR, self).__init__()

        self.b = b  ##! B Coefficients
        self.a = a  ##! A Coefficients
        self.zf = None  ##! Stored taps from previous samples

        assert len(b) == len(a) or (len(b) != len(a) and len(a) == 1)

    ## Not used
    def rxdown(self, data, meta=False):
        print "not used"

    ## Receive data from up
    def rxup(self, data, meta=False):

        # when self.zf is none, use zeros
        if self.zf is None:
            y, self.zf = lfilter(self.b, self.a, data, -1, [0]*(len(self.b)-1) )
        else:
            y, self.zf = lfilter(self.b, self.a, data, -1, self.zf)
            # print self.zf
        self.txdown(y)
