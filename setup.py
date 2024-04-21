import os
from setuptools import setup, find_packages

VERSION = "0.9.1"

INSTALL_REQUIRES = (
    'Orange3>=3.36',
    'orange3-network'
),

EXTRAS_REQUIRE = {
    'doc': ['sphinx', 'recommonmark', 'sphinx_rtd_theme'],
    'test': ['coverage'],
}

LONG_DESCRIPTION = open(os.path.join(os.path.dirname(__file__), 'README.pypi')).read()

ENTRY_POINTS = {
        'orange3.addon': ('pumice = orangecontrib.pumice', ),
        "orange.widgets": ("Pumice = orangecontrib.pumice.widgets", ),
        "orange.canvas.help": (
            'html-index = orangecontrib.prototypes.widgets:WIDGET_HELP_PATH'),
    }

setup(
    name="Orange3-Pumice",
    description="Educational widgets for project Pumice",
    license="BSD",
    version=VERSION,
    author='Bioinformatics Laboratory, FRI UL',
    author_email='contact@orange.biolab.si',
    url='https://github.com/biolab/orange3-pumice',
    keywords=(
        'orange3 add-on',
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={"orangecontrib.pumice.widgets": ["icons/*.svg"]},
    entry_points=ENTRY_POINTS,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    namespace_packages=['orangecontrib'],
    include_package_data=True,
    test_suite="orangecontrib.prototypes.tests.suite"
)
