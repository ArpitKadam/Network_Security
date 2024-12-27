from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """
    This function returns a list of requirements from requirements.txt.
    """
    requirement_list: List[str] = []
    try:
        with open('requirements.txt') as file:
            lines = file.readlines()
            requirement_list = [line.strip() for line in lines if line.strip() and line.strip() != '-e .']
    except Exception as e:
        print(f"Error: {e}")
    return requirement_list

setup(
    name='Network_Security',
    version='0.0.1',
    author='Arpit_Kadam',
    author_email='arpitkadam922@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
)
