from osibase import OsiBase

## This layer stores data in both up/down directions.
# Data received is available in the CaptureOsi.datafromdown or CaptureOsi.datafromup.
# Data is not passed through
class CaptureOsi(OsiBase):
    ## constructor
    # @param mode use ['extend','append']
    # @param passup Set to false to capture data in the down->up direction
    # @param passdown Set to false to capture data in the up->down direction
    #
    # In extend mode: when multiple lists are received, they are made into one list with no ability to detect which samples were part of which rx call
    #
    # In append mode: when multiple lists are received, a list of lists is made which allows you to see exactly which samples were part of which rx call
    def __init__(self, mode='extend', passup=False, passdown=False):
        super(CaptureOsi, self).__init__()
        self.datafromdown = [] ##! List of data received from down
        self.datafromup = []  ##! List of data received from up
        self.metafromdown = []  ##!List of metadata from down
        self.metafromup = []  ##! List of metadata from up
        self._rxsamples = 0  ##! Number of samples
        self._txsamples = 0  ##! Number of samples
        self._rxcalls = 0  ##! Number of calls
        self._txcalls = 0  ##! Number of calls
        self.passup = passup  ##! Should pass data up? if false data is captured
        self.passdown = passdown  ##! Should pass data down? if false data is captured

        assert mode == 'extend' or mode == 'append'

        if mode =='extend':
            self.extend_mode = True  ##! Boolean which determines mode
        else:
            self.extend_mode = False

    ## How many samples we have gotten through rxdown
    # @returns number of samples
    def rxcount(self):
        return self._rxsamples

    ## How many samples we have gotten through rxup
    # @returns number of samples
    def txcount(self):
        return self._txsamples

    ## Data from down
    def rxdown(self, data, meta=None):
        if not self.passup:
            self.metafromdown.append(meta)
            if self.extend_mode:
                self.datafromdown.extend(data)
            else:
                self.datafromdown.append(data)
        else:
            self.txup(data, meta)

        try:
            self._rxsamples += len(data)
        except TypeError:
            self._rxsamples += 1

        self._rxcalls += 1

    ## Data from up
    def rxup(self, data, meta=None):
        if not self.passdown:
            self.metafromup.append(meta)
            if self.extend_mode:
                self.datafromup.extend(data)
            else:
                self.datafromup.append(data)
        else:
            self.txdown(data, meta)

        try:
            self._txsamples += len(data)
        except TypeError:
            self._txsamples += 1

        self._txcalls += 1

    ## Empties CaptureOsi.datafromdown, and CaptureOsi.datafromup lists
    def dump(self):
        self.datafromdown = []
        self.datafromup = []


