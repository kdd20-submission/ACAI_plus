from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='acaiplus',
    version='0.1.0',
    packages=find_packages(),
    install_requires=required,
    url='',
    license='',
    author='',
    author_email='',
    description=''
)
