######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: April 17, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from setuptools import setup, find_packages

setup(
    name='LG-SIM-ENV',
    version='0.1.dev0',
    description='An Openai Environment for LG Car Simulator',
    url='',
    author='Munir Jojo-Verge',
    author_email='munirjojoverge@yahoo.es',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='autonomous driving simulation environment reinforcement learning (OpenAi Compatible Env)',
    packages=find_packages(exclude=['docs', 'scripts', 'tests']),
    install_requires=['gym', 'numpy', 'pygame', 'jupyter', 'matplotlib', 'pandas'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['scipy'],
        'deploy': ['pytest-runner', 'sphinx<1.7.3', 'sphinx_rtd_theme']
    },
    entry_points={
        'console_scripts': [],
    },
)

