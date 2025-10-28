from setuptools import setup, find_packages

setup(
    name="modules_FNA",        # The installable package name
    version="0.1",
    packages=find_packages(),   # Automatically finds all packages under modules_FNA/
    install_requires=[          # Optional: list dependencies
        # "numpy",
        # "torch",
        # "ray[tune]"
    ],
    python_requires=">=3.10",
)