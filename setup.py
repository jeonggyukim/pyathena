from setuptools import setup, find_packages

setup(
    name='pyathena',
    version='0.1',
    description='',
    author='',
    author_email='',
    python_requires='>=3.10',
    packages=find_packages(),
    extras_require={
        # Required only to regenerate the CHIANTI v11 derived tables
        # (ioneq, per-channel cool, per-ion atomic data) under
        # `pyathena.chemistry.tables.chianti_v11.*`. Runtime
        # consumers (IonCoolant, cool_chianti, ...) read the
        # pre-built .txt files and do not need ChiantiPy.
        # Also requires the CHIANTI data directory at
        # `$XUVTOP=~/Dropbox/Projects/CHIANTI_db` (or equivalent).
        'tables': ['ChiantiPy>=0.16'],
        # Documentation build (Sphinx + jupyter-book 1.x stack).
        'docs': [
            'jupyter-book<2',
            'sphinx',
            'myst-parser',
            'sphinx-autoapi',
            'furo',
        ],
    },
)
