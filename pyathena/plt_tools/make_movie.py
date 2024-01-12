import io, sys
import subprocess
import base64
from IPython.display import HTML

def make_movie(fname_glob, fname_out, fps_in=15, fps_out=15):
    """(wrapper) function to create an mp4 movie from files matching a glob
    pattern.

    Parameters
    ----------
    fname_glob : string (glob)
        Glob pattern including wildcard
    fname_out : string
        Name of mp4 output file
    fps_in : int
        Frames per second for input
    fps_output : int
        Frames per second for output

    Example
    -------
    To force the frame rate of the input file (valid for raw formats only) to 1
    fps and the frame rate of the output file to 24 fps

    >>> make_movie('a.????.png', 'a.mp4', fps_in=1, fps_out=24)

    """

    cmd = ['ffmpeg',
           '-y', # override existing file
           '-r', str(fps_in),
           '-f', 'image2',
           '-pattern_type', 'glob',
           '-i', fname_glob,
           '-r', str(fps_out),
           '-pix_fmt', 'yuv420p',
           '-vcodec', 'libx264',
           '-vf', r'scale=trunc\(iw/2\)*2:trunc\(ih/2\)*2',
           '-f', 'mp4', fname_out]

    print('[make_movie]: ffmpeg command:')
    print('{0:s}'.format(' '.join(cmd)))

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)

        # ret = subprocess.check_call(cmd)
        # df = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        # output, err = df.communicate()
        print('[make_movie]: Successful execution.')
        print('[make_movie]: Movie:')
        print('{0:s}'.format(fname_out))
        return True
    except subprocess.CalledProcessError as e:
        print("\x1b[31m[make_movie]: subprocess.check_output returned:\x1b[0m")
        print(str(e.output, "utf-8"))
        return False


def display_movie(filename):

    video = io.open(filename, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>'''.format(encoded.decode('ascii')))
