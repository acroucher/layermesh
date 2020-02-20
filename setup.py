from setuptools import setup

setup(
    name='layermesh',
    version='0.1',
    description='Library for layered computational meshes',
    url='http://github.com/acroucher/layermesh',
    author='Adrian Croucher',
    author_email='a.croucher@auckland.ac.nz',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent"],
    install_requires=['numpy']
)
