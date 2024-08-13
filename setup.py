from setuptools import setup, find_packages

setup(
    name='TERNTEQP',
    version='0.1',
    packages=find_packages(),  # Automatically find and include all packages
    include_package_data=True,
    install_requires=["teqp","numpy","scipy","mpltern","matplotlib"],
)