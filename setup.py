"""Setup: pip install . или pip install -e ."""
from setuptools import setup, find_packages

setup(
    name="newsbot",
    version="1.0.0",
    packages=find_packages(),
    package_data={
        # __main__.py — нужен для `python -m newsbot`
        "newsbot": ["__main__.py"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "feedparser>=6.0.10",
        "httpx>=0.27.0",
        "PyYAML>=6.0.1",
    ],
)