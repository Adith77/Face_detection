from setuptools import setup, find_packages

setup(
    name='face_recognition',
    version='1.3.0',
    packages=find_packages(),
    install_requires=[
        'Click>=6.0',
        'dlib-bin',  # Use dlib-bin instead of dlib
        'face_recognition_models>=0.3.0',
        'numpy',
        'Pillow',
    ],
    entry_points='''
        [console_scripts]
        face_recognition=face_recognition.cli:cli
    ''',
)
