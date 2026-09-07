"""
Copyright (c) 2025-2026 KylinSoft Co., Ltd.

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from setuptools import setup, find_packages

setup(
    name='fgds',
    version='0.1.2',
    package_dir={"fgds": "fgds", "fgds_backend": "fgds_backend"},
    author='kuangkai',
    author_email='kuangkai@kylinos.cn',
    description='External logging backend implementation for LMCache',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Storage-and-OS-for-AI/fgds',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

