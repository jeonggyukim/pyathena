import io
import subprocess
import base64
from IPython.display import HTML

def make_mp4(fglob, fout, fps_in=10, fps_out=20):
    # To force the frame rate of the input file (valid for raw formats only) to 1 fps and the frame rate of the output file to 24 fps:

    cmd = ['ffmpeg',
           '-y', # override existing file
           '-r', str(fps_in),
           '-f', 'image2',
           '-pattern_type', 'glob',
           '-i', fglob,
           '-r', str(fps_out),
           '-pix_fmt', 'yuv420p',
           '-vcodec', 'libx264',
           '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
           '-f', 'mp4', fout]

    print('[make_mp4]: ffmpeg command')
    print('{0:s}'.format(' '.join(cmd)))
    
    return subprocess.call(cmd)

def display_mp4(filename):

    video = io.open(filename, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>'''.format(encoded.decode('ascii')))
