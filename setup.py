from setuptools import setup

with open("VERSION") as file:
    version = file.readline()

requirements = [
        "numpy>=1.18.5",
        "matplotlib>=3.3.2",
        "scipy>=1.5.4",
        "re>=2.2.1"
        ]

setup(
        name="dsc_thermo",
        package_data = {"dsc_thermo": ["data/*"],
        version = version,
        install_requires=requirements
        )

