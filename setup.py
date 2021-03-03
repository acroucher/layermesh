import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='layermesh',
    version='0.2',
    description='Library for layered computational meshes',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/acroucher/layermesh',
    author='Adrian Croucher',
    author_email='a.croucher@auckland.ac.nz',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent"],
    install_requires=['numpy', 'scipy', 'h5py', 'meshio', 'matplotlib']
)
