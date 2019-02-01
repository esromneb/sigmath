
## Single ended filter chain element
#
class FilterElement(object):

    ## Constructor
    def __init__(self):
        self.nextelement = None  ##! points at next in chain
        self.name = "noname"  ##! nicename for printing

    ## Call to input data into the filter
    def input(self, data, meta=None):
        return self.down.rxup(data, meta)

    def output(self, data, meta=None):
        return self.nextelement.input(data, meta)

    ## Call this regularly on blocks which impliment it
    def tick(self):
        pass

    ## String classes together with this
    def set_next(self, n):
        self.nextelement = n
