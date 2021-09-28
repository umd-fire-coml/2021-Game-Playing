# Python Automated Testing Example

![example workflow](https://github.com/umd-fire-coml/example-automated-tests/actions/workflows/run_all_tests.yaml/badge.svg)

This package contains the following files:

```bash
├── src
│   ├── hello.py
│   ├── goodbye.py
├── test
│   ├── test_sanity.py
│   ├── test_hello.py
│   ├── test_goodbye.py
├── notebooks
│   ├── Example.ipynb
├── .github
│   ├── workflows
│       ├── main.yaml
├── requirements.txt
├── requirements-test.txt
└── .gitignore
```

## Source Directory

The src directory contains all the source code files for your project. In this example, there are two files - hello.py which contains the Hello class and goodbye.py which contains the Goodbye class.

## Test Directory

The test directory contains all the unit tests for our source code files (test_hello.py, test_goodbye.py) as well as a test_sanity.py that just makes sure our tests work as expected. 

## Notebooks Directory

The notebooks directory contains an example interactive python notebook that uses the source files.

## .github Directory

The .github/workflows directory contains the different workflows that we want to run using GitHub Actions. The main.yaml file runs all of our tests as separate jobs any time we push, create a pull request, or manually run on GitHub.

## Requirements

The requirements.txt file and requirements-test.txt file contain the required packages for the code and tests respectively.

## Testing Locally

To run all tests locally, make sure pytest is installed. Then, run
```bash
python -m pytest
```
in the root directory.

To run a single test, make sure pytest is installed. Them, run
```bash
python -m pytest test/test_file_you_want_to_run.py
```

## Testing On GitHub Actions

To view the tests that have run and their logs,

1. Go to your GitHub Repository
2. Click on Actions
3. View the most recent run of your workflow or manually run your workflow

To add a new test to the main.yaml file, add a new job with the following format
```yaml
  Your_Job_Name:
    runs-on: ubuntu-latest
    container: python
    steps:
      - uses: actions/checkout@v2
      - name: Your_Step_Name
        run: |
          pip install -r requirements.txt
          pip install -r test-requirements.txt
          python -m pytest test/test_file_you_want_to_run.py
``` 
