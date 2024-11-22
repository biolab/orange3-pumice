import os
from setuptools import setup, find_packages
from setuptools.command.install import install

VERSION = "0.9.1"

INSTALL_REQUIRES = (
    'Orange3>=3.37',
    'orange3-network',
    'orange3-imageanalytics',
    "orange-canvas-core >=0.2.2",
    "orange-widget-base >=4.23.0",
),

EXTRAS_REQUIRE = {
    'doc': ['sphinx', 'recommonmark', 'sphinx_rtd_theme'],
    'test': ['coverage'],
}

SETUP_REQUIRES = [
    'trubar>=0.3.3',
]

LONG_DESCRIPTION = open(os.path.join(os.path.dirname(__file__), 'README.pypi')).read()

ENTRY_POINTS = {
        'orange3.addon': ('pumice = orangecontrib.pumice', ),
        "orange.widgets": ("Pumice = orangecontrib.pumice.widgets", ),
        "orange.canvas.help": (
            'html-index = orangecontrib.prototypes.widgets:WIDGET_HELP_PATH'),
    }

class InstallMultilingualCommand(install):
    def run(self):
        install.run(self)
        self.compile_to_multilingual()

    def compile_to_multilingual(self):
        from trubar import translate

        package_dir = os.path.dirname(os.path.abspath(__file__))
        translate(
            "msgs.jaml",
            source_dir=os.path.join(self.install_lib, "orangecontrib", "pumice"),
            config_file=os.path.join(package_dir, "i18n", "trubar-config.yaml"))


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
    setup_requires=SETUP_REQUIRES,
    cmdclass={
        'install': InstallMultilingualCommand,
    },
    extras_require=EXTRAS_REQUIRE,
    namespace_packages=['orangecontrib'],
    include_package_data=True,
    test_suite="orangecontrib.prototypes.tests.suite"
)
