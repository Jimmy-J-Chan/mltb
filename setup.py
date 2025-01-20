import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="mltb",
    version="2024.08.04",
    author="Jimmy Chan",
    author_email="jimmychan0507@gmail.com",
    # packages=["mltb",
    #           "model_selection",
    #           "utils"],
    description="Machine Learning ToolBox",
    long_description=description,
    long_description_content_type="text/markdown",
    url="",
    license='MIT',
    python_requires='>=3.10',
    install_requires=['pandas',
                      'numpy',
                      'matplotlib',
                      'PyYAML',
                      'polars',
                      'pyarrow',
                      'scikit-learn',
                      'lightgbm',
                      'catboost',
                      'xgboost',
                      'optuna'
                      ]
)