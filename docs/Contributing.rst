Contributing
=======================================

Bugs reports, feature requests and questions
---------------------------------------------

Please use the `Issue Tracker <https://github.com/WMD-group/PDynA/issues>`_ to report bugs or
request new features. Contributions to extend this package are very welcome!

Development Setup
-----------------

To set up a development environment:

.. code-block:: bash

   git clone https://github.com/WMD-group/PDynA.git
   cd PDynA
   pip install -e ".[test]"

Running Tests
-------------

The test suite uses ``pytest``. To run all tests:

.. code-block:: bash

   pytest tests/ -v

To run tests with coverage reporting:

.. code-block:: bash

   pytest tests/ -v --cov=pdyna --cov-report=term-missing

Continuous Integration
----------------------

Pull requests are automatically tested using GitHub Actions. The CI pipeline:

- Runs the test suite on Python 3.9, 3.10, and 3.11
- Performs basic code linting with ruff

Please ensure all tests pass before submitting a pull request.

Code Style
----------

- Follow PEP 8 guidelines for Python code
- Use descriptive variable and function names
- Add docstrings to new functions and classes
