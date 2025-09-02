from setuptools import setup, find_packages

setup(
    name="customer360",
    version="0.1.1",
    description="Customer 360 package (dedupe, unit freezing, rollups, scoring, Streamlit UI)",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=2.1",
        "numpy>=1.26",
        "rapidfuzz>=3.9",
        "python-dateutil>=2.9",
        "openpyxl>=3.1",
        "xlsxwriter>=3.2",
    ],
)
