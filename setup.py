from setuptools import find_packages, setup

setup(
    name='memento-de',
    version='0.1.2',
    description='Hypothesis testing for scRNA-seq',
    url='https://github.com/yelabucsf/scrna-parameter-estimation.git',
    author='Min Cheol Kim',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'anndata>=0.8',
        'joblib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'statsmodels',
    ],
)
