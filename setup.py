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
        # Запиненные версии (pip show в текущем окружении):
        #   feedparser==6.0.12
        #   httpx==0.28.1
        #   PyYAML==6.0.3
        "feedparser==6.0.12",
        "httpx==0.28.1",
        "PyYAML==6.0.3",
    ],
)