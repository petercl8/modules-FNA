from setuptools import setup, find_packages

setup(
    name="modules_FNA",
    version="0.1",
    packages=find_packages(include=["modules_FNA", "modules_FNA.*"]),
    install_requires=[
        # Add external dependencies here, e.g. "numpy", "torch"
    ],
)
