from pathlib import Path
from setuptools import setup, find_packages


package_dir = 'topic_analysis'
root = Path(__file__).parent.resolve()


# Read in package meta from about.py
about_path = root / package_dir / 'about.py'
with about_path.open('r', encoding='utf8') as f:
    about = {}
    exec(f.read(), about)


# Get readme
readme_path = root / 'README.md'
with readme_path.open('r', encoding='utf8') as f:
    readme = f.read()


install_requires = [
    'adjustText',
    'joblib',
    'matplotlib',
    'numpy>=1.16',
    'pandas',
    'progressbar2',
    'psutil',
    'pystemmer',
    'scipy',
    'scikit-learn',
    'spacy',
    'spacy-lookups-data',
    'symspellpy',
    'unidecode'
]
test_requires = ['pytest']


def have_compiler():
    """
    checks for the existence of a compiler by compiling a small C source file
    source: https://charlesleifer.com/blog/misadventures-in-python-packaging-optional-c-extensions/
    """
    from distutils.ccompiler import new_compiler
    from distutils.errors import CompileError
    from distutils.errors import DistutilsExecError
    import os
    import tempfile
    import warnings
    fd, fname = tempfile.mkstemp('.c', text=True)
    f = os.fdopen(fd, 'w')
    f.write('int main(int argc, char** argv) { return 0; }')
    f.close()
    compiler = new_compiler()
    try:
        compiler.compile([fname])
    except (CompileError, DistutilsExecError):
        warnings.warn('compiler not installed')
        return False
    except Exception as exc:
        warnings.warn('unexpected error encountered while testing if compiler '
                      'available: %s' % exc)
        return False
    else:
        return True


if not have_compiler():
    install_requires.remove('pystemmer')
    install_requires += ['nltk']


setup(
    name=about['__title__'],
    description=about['__summary__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__email__'],
    url=about['__uri__'],
    version=about['__version__'],
    license=about['__license__'],
    packages=find_packages(exclude=('tests*',)),
    install_requires=install_requires,
    test_requires=test_requires,
    zip_safe=True,
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Topic :: Text Processing :: General'
    ]
)

# install spacy languages
import os
import sys
os.system(f'{sys.executable} -m spacy download nl_core_news_sm --user')
os.system(f'{sys.executable} -m spacy download en_core_web_sm --user')