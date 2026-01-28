from setuptools import setup, find_packages

setup(
    name='fgds',
    version='0.1.2',
    packages=find_packages(),
    author='kuangkai',
    author_email='kuangkai@kylinos.cn',
    description='The python module of fgds api',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Storage-and-OS-for-AI/fgds',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)

