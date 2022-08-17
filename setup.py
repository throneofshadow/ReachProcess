from setuptools import setup

setup(
    name='ReachProcess',
    version='0.1.0',
    description='A python library for extracting and processing video and experimental data from the'
                'ReachMaster experimental paradigm. The processes used in this function rely on a pre-trained'
                'DeepLabCut network in order to extract positional predictions from subjects within the video.',
    url='https://github.com/throneofshadow/ReachProcess',
    author='Brett Nelson',
    author_email='bnelson@lbl.gov',
    license='BSD-3-Clause-LBNL',
    packages=['ReachSample'],
    install_requires=['cv2', 'vidgear', 'pdb','numpy','glob','collections','os','glob', 'tqdm', 'pickle', 'scipy'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)