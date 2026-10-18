"""Setup: pip install . или pip install -e ."""
from pathlib import Path
import re

_version_file = Path(__file__).parent / "newsbot" / "version.py"
_version = re.search(r'__version__\s*=\s*"([^"]+)"', _version_file.read_text(encoding="utf-8")).group(1)

from setuptools import setup, find_packages

setup(
    name="newsbot",
    version=_version,
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