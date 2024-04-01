from setuptools import setup, find_packages

INSTALL_REQUIRES = (
    'Orange3>=3.35',
    'orange3-network'
),

setup(
    name="Orange3-Pumice",
    packages=find_packages(),
    package_data={"orangecontrib.pumice.widgets": ["icons/*.svg"]},
    entry_points={
        'orange3.addon': ('educational = orangecontrib.educational', ),
        "orange.widgets": ("Pumice = orangecontrib.pumice.widgets", )
    },
    install_requires=INSTALL_REQUIRES,
)