from interface import Interface, implements



class TickableInterface(Interface):
    def tick(self):
        pass



# class

class OsiBaseInterfaceTop(Interface):

    ## receive from upper layer  VIRTUAL
    def rxup(self, data, meta=None):
        print "warning", self, "does not impliment rxup()"
        return self.txdown(data, meta)

    ## transmit to the upper layer
    #  @param data same as txdown()
    # #  @param meta is meta data usually a {}
    # def txup(self, data, meta=None):
    #     return self.up.rxdown(data, meta)

class OsiBaseInterfaceBottom(Interface):
    # def txdown(self, data, meta=None):
    #     return self.down.rxup(data, meta)

    ## receive from the lower layer  VIRTUAL
    #  @param meta is meta data usually a {}z
    def rxdown(self, data, meta=None):
        print "warning", self, "does not impliment rxdown()"
        return self.txup(data, meta)

class OsiBaseInterface(OsiBaseInterfaceBottom, OsiBaseInterfaceTop, TickableInterface):
    def set_parent(self, parent):
        pass

    def insert_above(self, newparent):
        pass

    def insert_below(self, newchild):
        pass

    pass




