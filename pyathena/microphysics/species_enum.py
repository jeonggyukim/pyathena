from aenum import IntEnum, NoAlias


class EnumAtom(IntEnum):

    H = 1
    He = 2
    Li = 3
    Be = 4
    B = 5
    C = 6
    N = 7
    O = 8
    F = 9
    Ne = 10
    Na = 11
    Mg = 12
    Al = 13
    Si = 14
    P = 15
    S = 16
    Cl = 17
    Ar = 18
    K = 19
    Ca = 20
    Sc = 21
    Ti = 22
    V = 23
    Cr = 24
    Mn = 25
    Fe = 26
    Co = 27
    Ni = 28
    Cu = 29
    Zn = 30


class EnumCtypes(IntEnum):
    """A ctypes-compatible IntEnum superclass."""
    @classmethod
    def from_param(cls, obj):
        return int(obj)


class EnumLineCoolElem(EnumCtypes):
    """
    IntEnum for LineCool Elements
    Should be identical to enum in c and c++ header files
    """

    # https://stackoverflow.com/questions/31537316/python-enums-with-duplicate-values
    _settings_ = NoAlias

    # 5 level
    NI = 0
    NII = 1
    OI = 2
    OII = 3
    OIII = 4
    NeIII = 5
    SII = 6
    SIII = 7
    CII = 8
    CIII = 9

    # 2 level
    NIII = 10
    NeII = 11
    SIV = 12




class EnumLineCoolTransition(EnumCtypes):
    """
    IntEnum for LineCool 5 level transition
    Should be identical to enum in c and c++ header files
    """

    T01 = 0
    T02 = 1
    T03 = 2
    T04 = 3
    T12 = 4
    T13 = 5
    T14 = 6
    T23 = 7
    T24 = 8
    T34 = 9
