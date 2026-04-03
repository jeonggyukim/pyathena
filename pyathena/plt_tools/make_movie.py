import io, sys
import subprocess
import base64
from IPython.display import HTML

def make_movie(fname_glob, fname_out, fps_in=15, fps_out=15):
    """Create an mp4 movie from image files matching a glob pattern.

    Requires ``ffmpeg`` to be installed and available on ``PATH``.
    Output is encoded as H.264 with yuv420p pixel format for broad
    compatibility.

    Parameters
    ----------
    fname_glob : str
        Glob pattern matching the input image files (e.g. ``'frame.????.png'``).
    fname_out : str
        Path of the output ``.mp4`` file. Overwritten if it already exists.
    fps_in : int, optional
        Frame rate of the input image sequence. Default is 15.
    fps_out : int, optional
        Frame rate of the output video. Default is 15.

    Returns
    -------
    bool
        ``True`` if ffmpeg completed successfully, ``False`` otherwise.

    Examples
    --------
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
    """Display an mp4 video inline in a Jupyter notebook.

    Parameters
    ----------
    filename : str
        Path to the ``.mp4`` file.

    Returns
    -------
    IPython.display.HTML
        An HTML video element for inline playback.
    """
    video = io.open(filename, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>'''.format(encoded.decode('ascii')))
