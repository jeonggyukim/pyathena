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
        string = 'PAR_DUMP'
        while True:         # Read until 'PAR_DUMP'
            line = f.readline()
            if not line:
                print('{0} not found in {1}'.format(string,filename))
                raise IOError('{0} not found in {1}'.format(string,filename))
            if string in line:
                break

        while True:        # Read and save until 'PAR_DUMP'
            line = f.readline().strip()
            if 'PAR_DUMP' in line:
                break
            lines.append(line)

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

    block.remove('comment')    # remove comment block
    o = {}
    for b in block:
        o.setdefault(b, {})

    # Add keys and values to each block
    for l in lines:
        b = reblock.match(l)
        p = reparam.match(l)
        if b is not None:
            bstr = b.group().strip()[1:-1]
            if o.has_key(bstr):
                bname = bstr # bname is valid block
            else:
                bname = None
        elif p is not None and bname is not None:
            pname = l.split()[0]
            value = l.split()[2]
            
            # See if pname contains space
            if value == '=':
                pname = " ".join(l.split()[0:2])
                value = l.split()[3]
                
            # Evaluate if value is floating point number (or real number) or integer or string
            # Too complicated...there must be a better way...
            if re.match(r'^[+-]?\d+\.\d+[eE][+-]?\d+$',value) or \
               re.match(r'^[+-]?\d+[eE][+-]?\d+$',value) or \
               re.match(r'^[+-]?\d+\.\d*$',value):
                o[bname][pname] = float(value)
            elif re.match(r'^-?[0-9]+$',value):
                o[bname][pname] = int(value)
#                o[bname][l.split()[0]]=float(value)
            else:
                o[bname][pname] = value

    par = collections.namedtuple('par', o.keys())(**o)
    
    return par
