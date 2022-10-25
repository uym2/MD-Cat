from setuptools import setup, find_packages
import emd
from os import walk, listdir
from os.path import join,normpath,isfile

param = {
    'name': emd.PROGRAM_NAME,
    'version': emd.PROGRAM_VERSION,
    'description': emd.PROGRAM_DESCRIPTION,
    'author': emd.PROGRAM_AUTHOR,
    'license': emd.PROGRAM_LICENSE,
    'packages': find_packages(),
    'include_package_data': True,
    'scripts' : ['md_cat.py','simulate.py'],
    'zip_safe': True,
    'install_requires': ['treeswift','scipy>=1.3.1','bitsets','numpy>=1.18.5','jenkspy','mosek','cvxpy','cvxopt'],
    'keywords': 'Phylogenetics Evolution Biology',
    'long_description': """A Python implementation of the MD-Cat algorithm""",
    'classifiers': ["Environment :: Console",
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: GNU General Public License (GPL)",
                    "Natural Language :: English",
                    "Operating System :: OS Independent",
                    "Programming Language :: Python",
                    "Topic :: Scientific/Engineering :: Bio-Informatics",
                    ],
    }
    
setup(**param)
