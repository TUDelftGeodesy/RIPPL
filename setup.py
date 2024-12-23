from setuptools import setup, find_packages

# https://packaging.python.org/guides/single-sourcing-package-version/
version_info = {}
with open("rippl/version.py") as file:
    exec(file.read(), version_info)

# defining dependencies
with open("requirements.txt") as fp:
    install_requires = fp.read()

setup(
    name="RIPPL",
    version=version_info["__version__"],
    author="Gert Mulder",
    author_email="g.mulder-1@tudelft.nl",
    url="https://github.com/TUDelftGeodesy/RIPPL",
    packages=find_packages(),
    license="GNU LGPLv3",
    description="[R]adar [I]nterferometric [P]arallel [P]rocessing [L]ab",
    setup_requires=["numpy"],
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU LGPLv3",
        "Programming Language :: Python :: 3",
    ],
    keywords=[
        "InSAR",
        "Sentinel-1",
        "SAR",
        "Interferometry",
        "Parallel processing",
        "Stack processing",
        "Remotes Sensing",
    ],
)
