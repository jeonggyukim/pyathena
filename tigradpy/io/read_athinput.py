import re
import collections

def read_athinput(filename, verbose=False):
    """
    Function to read athinput and configure block from simulation log
    
    Parameters
    ----------
    filename : string
        Name of the file to open, including extension
    verbose : bool
    
    Returns
    -------
    par : namedtuple
        Each item is a dictionary for input block
    """

    if verbose:
        print('[read_par]: Reading params from {0}'.format(filename))

    lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    DUMP = False
    for i, line in enumerate(lines):
        if 'PAR_DUMP' in line:
            DUMP = True
            break

    if DUMP: # from simulation log
        flag = True
        for i, line in enumerate(lines):
            if 'PAR_DUMP' in line:
                if flag:
                    istart = i
                    flag = False
                else:
                    iend = i
    else: # from restart
        for i, line in enumerate(lines):
            if '<comment>' in line:
                    istart = i
            if '<par_end>' in line:
                    iend = i
                    
    lines = lines[istart:iend]
    
    # Parse lines
    reblock = re.compile(r"<\w+>\s*")
    ##  reparam=re.compile(r"[-\[\]\w]+\s*=")
    # To deal with space (such as star particles in <configure>)
    reparam = re.compile(r"[-\[\]\w]+[\s*[-\[\]\w]*]*\s*=")

    # Find blocks first
    block = []
    for l in lines:
        b = reblock.match(l)
        if b is not None:
            block.append(b.group().strip()[1:-1])

    # remove comment block
    block.remove('comment')

    o = {}
    for b in block:
        o.setdefault(b, {})

    # Add keys and values to each block
    for l in lines:
        b = reblock.match(l)
        p = reparam.match(l)
        if b is not None:
            bstr = b.group().strip()[1:-1]
            if bstr in o:
                bname = bstr # bname is valid block
            else:
                bname = None
        elif p is not None and bname is not None:
            lsplit = l.split()
            i1_found=False
            for i, lsplit_ in enumerate(lsplit):
                if lsplit_ == '=':
                    i0 = i
                    break

            pname = '_'.join(lsplit[:i0])
            value = lsplit[i0+1]

            # Evaluate if value is floating point number (or real number) or integer or string
            # Too complicated...there must be a better way...
            if re.match(r'^[+-]?\d*\.\d*[eE][+-]?\d+$',value) or \
               re.match(r'^[+-]?\d+[eE][+-]?\d+$',value) or \
               re.match(r'^[+-]?\d+\.\d*$',value):
                o[bname][pname] = float(value)
            elif re.match(r'^-?[0-9]+$',value):
                o[bname][pname] = int(value)
#                o[bname][l.split()[0]]=float(value)
            else:
                o[bname][pname] = value

    return o

    ## Convert to namedtuple
    # par = collections.namedtuple('par', o.keys())(**o)
    # return par
