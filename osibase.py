import numpy as np

from siginterface import OsiBaseInterface, OsiBaseInterfaceTop, OsiBaseInterfaceBottom
from interface import implements


## A Basic block for building transceiver chains
#
#  Blessed are the meek: for they shall inherit this class
class OsiBase(implements(OsiBaseInterface)):
    ## Constructor
    ##  ctons
    def __init__(self):
        self.up = None  ##! Points at next in rx chain
        self.down = None  ##! Points at next in tx chain
        self.name = "noname"  ##! Nicename for printing

    ## Traditionally the "TX" direction
    #  @param data rf: (list,np.array) data: (byte strings)
    #  transmit to the lower layer
    def txdown(self, data, meta=None):
        return self.down.rxup(data, meta)

    ## Transmit to the upper layer
    #  @param data same as txdown()
    #  @param meta is meta data usually a {}
    def txup(self, data, meta=None):
        return self.up.rxdown(data, meta)

    ## Receive from the lower layer VIRTUAL
    #  @param meta is meta data usually a {}
    def rxdown(self, data, meta=None):
        print "warning", self, "does not implement rxdown()"
        return self.txup(data, meta)

    ## Receive from upper layer VIRTUAL
    def rxup(self, data, meta=None):
        print "warning", self, "does not implement rxup()"
        return self.txdown(data, meta)

    ## Call this regularly on blocks which implement it. Dummy for now
    def tick(self):
        pass

    ## String classes together with this, Set the argument to become the parent of the object who calls this method.
    def set_parent(self, parent):
        self.up = parent
        parent.down = self
        return parent  # this allows for stringing on one line

    ## Insert a parent to the object who calls this method
    def insert_above(self, newparent):
        oldparent = self.up
        self.set_parent(newparent).set_parent(oldparent)

    ## Insert a child to the object who calls this method
    def insert_below(self, newchild):
        oldchild = self.down
        oldchild.set_parent(newchild).set_parent(self)
