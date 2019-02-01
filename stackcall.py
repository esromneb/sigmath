from captureosi import CaptureOsi
from osibase import OsiBase

## Wraps a class and makes it easier to call data one-off.
# Adds a capture osi on either side of a class and allows for an easy calling of it
class StackCall(OsiBase):
    ## Constructor. Start from parent, the layers are: above->wrap->below
    # @param wrap The class to be wrapped
    def __init__(self, wrap, mode='extend'):
        self.wrap = wrap  ##! The object to be wrapped
        self.capturemode = mode  ##! Capture mode: extend or append, default extend
        self.above = CaptureOsi(self.capturemode)  ##! CaptureOsi object above wrap
        self.below = CaptureOsi(self.capturemode)  ##! CaptureOsi object below wrap

        self.wrap.set_parent(self.above)
        self.below.set_parent(self.wrap)

    ## Call the wrapped up->down (tx)
    # @returns the data
    def rxup(self, data, meta=False):
        self.wrap.rxup(data)
        if self.capturemode == 'extend':
            dout = self.below.datafromup
        else:
            dout = self.below.datafromup[0]
        self.below.dump()
        return dout

    ## Call the wrapped down->up (rx)
    # @returns the data
    def rxdown(self, data, meta=False):
        self.wrap.rxdown(data)
        if self.capturemode == 'extend':
            dout = self.above.datafromdown
        else:
            dout = self.above.datafromdown[0]
        self.above.dump()
        return dout
