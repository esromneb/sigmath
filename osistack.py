## This is used for gluing objects of type OsiBase only (currently in the wrong file: sigosi.py)
#
# It does not inehrit that type of object.  However it can be bound with up/down pointers like
# other classes of type OsiBase
class OsiStack(object):
    ## Constructor
    def __init__(self):
        self.stack = []  ##! Stack list

    def applyname(self, name, obj):
        setattr(self, name, obj)  # apply to OsiStack
        for s in self.stack:
            if s != obj:
                setattr(s, name, obj)

    def applyallnames(self):
        for s in self.stack:
            if hasattr(s, 'stackname'):
                self.applyname(s.stackname(), s)

    ## Append p to the end of stack
    # @param p object to be added into stack
    def add(self, p):
        idx = len(self.stack)
        self.stack.append(p)
        if idx > 0:
            self.stack[idx-1].set_parent(self.stack[idx])
        self.applyallnames()

    ## Append p to the start of the stack
    # @param p object to be added into stack
    def addbottom(self, p):
        idx = len(self.stack)
        self.stack.insert(0,p)
        # self.stack.append(p)
        if idx > 0:
            self.stack[0].set_parent(self.stack[1])
        self.applyallnames()

    def tick(self, count = 1):
        for _ in range(count):
            for s in self.stack:
                s.tick()

    ## Get stack name
    # @returns stack name in string
    def _name(self):
        return str(self.stack)

    def __getattr__(self, name):
        if name == 'name':
            return self._name()
        elif name == 'log':
            print "warning might not set all logs"
            return self.stack[0].log
        else:
            return self.__dict__[name]

    def set_parent(self, p):
        assert len(self.stack)
        self.stack[len(self.stack)-1].set_parent(p)

    def __setattr__(self, name, value):
        # print name, value
        if name == 'down':
            assert len(self.stack)
            self.stack[0].down = value
        elif name == 'up':
            self.stack[len(self.stack)-1].up = value
        else:
            self.__dict__[name] = value
        # super(OsiStack, self).__setattr__(key, value)
        # self[key] = value

    # def txdown(self, data):
    #     return self.down.rxup(data)

    # transmit to the upper later
    # def txup(self, data):
    #     return self.up.rxdown(data)

    ## Receive from the lower layer VIRTUAL
    def rxdown(self, data):
        assert len(self.stack)
        return self.stack[0].rxdown(data)

    ## Receive from upper later VIRTUAL
    def rxup(self, data):
        return self.stack[len(self.stack)-1].rxup(data)