from setuptools import setup

setup(
    name="Rec-System",
    packages=["Recommendation"],
    package_data={"Recommendation": ["icons/*.svg"]},
    entry_points={"orange.widgets": "Rec-System = Recommendation"},
)