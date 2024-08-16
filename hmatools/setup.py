from setuptools import setup, find_packages

setup(
    name="HMATools",
    version="0.1",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=["argparse", "pandas", "seaborn", "matplotlib"],
    entry_points={
        "console_scripts": [
            "postprocess=postprocparser:main",
            "plot=plotparser:main",
        ],
    },
    test_suite="python/tests",
)
