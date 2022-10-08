from setuptools import setup, find_packages

tests_require = [
    'pytest',
    'pytest-tornasync',
    'mypy'
]
setup(
    name='kerasserver',
    version='0.6.0',
    author_email='andrey@onepanel.io',
    license='TODO',
    url='TODO',
    description='Model Server implementation for Keras. ' +
    'Not intended for use outside KFServing Frameworks Images',
    long_description=open('README.md').read(),
    python_requires='>3.6',
    packages=find_packages("kerasserver"),
    install_requires=[
        "kfserving==0.6.0",
        "tensorflow==1.13.1",
        'numpy',
        'cython',
        'pyyaml',
        'keras==2.1.0',
        'scikit-image',
        'Pillow'
    ],
    tests_require=tests_require,
    extras_require={'test': tests_require}
)
