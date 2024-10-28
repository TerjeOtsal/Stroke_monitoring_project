# setup.py
from setuptools import setup, find_packages

setup(
    name='webapp',                   # Package name
    version='0.1.0',                 # Package version
    packages=find_packages(),        # Automatically find packages in the `webapp` folder
    include_package_data=True,       # Include static and template files
    install_requires=[
        'Flask>=2.0.0',              # Define dependencies (Flask in this case)
    ],
    entry_points={
        'console_scripts': [
            'webapp=webapp.app:app.run',   # Create a console script to run the app
        ],
    },
    package_data={
        '': ['static/*', 'templates/*'],   # Include static and templates files in the package
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A sample installable Flask app",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
