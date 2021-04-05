# Command Input Output Testing Tool in Python

Tool to test the Command Input/Output.

It is a single Python script file (`ciot.py`), that can be added to path/renamed ...

```shell
$ python -m ciot                                                                                                                                                                                                                                                                                                                                              Mon 05 Apr 2021 18:54:07 UTC
usage: ciot [-h] [-L LOG_LEVEL] {parse,exec} ...

Command Input Output Testing Tool (ciot) TBD

optional arguments:
  -h, --help            show this help message and exit
  -L LOG_LEVEL, --log-level LOG_LEVEL
                        Set log level (DEBUG|INFO|WARNING|ERROR)

Sub-Commands:
  {parse,exec}
    parse               Parse and print the mini hw scenario
    exec                Execute the unit file
```

### Requirements

- Python 3.7
- [``PyYaml``](https://pypi.org/project/PyYAML/) - to parse the `yml`, `yaml` templates (optional)
- [``JUnitParser``](https://pypi.org/project/junitparser/) - to write the JUNIT reports (optional)

Both dependencies are optional, without ``PyYaml`` you would need to use only `json` unit templates.
Without ``JUnitParser`` you would not be able to generate
the [JUNIT CML Report](https://docs.gitlab.com/ee/ci/unit_test_reports.html)

### Installation:

```shell
pip install --user junitparser
pip install --user PyYAML

wget https://github.com/pestanko/pyciot/blob/master/ciot.py
```

Or clone the repository:

```shell
git clone https://github.com/pestanko/pyciot.git
```

For development purposes this repository is unsing the [`Poetry`](https://python-poetry.org/) to manage Virtual Envs.
Since this package has only 2 dependencies, it is not required to use it.

```shell
cd pyciot
poetry install --no-root 
poetry shell
```


### Example Usage:

These examples will work from within the repository.

```shell
# Execute example scenario on /usr/bin/echo command
python ciot.py -Ldebug exec -T examples/echocat/tests -U examples/echocat/tests/unit-echo.yml -C /usr/bin/echo
# Scenario should pass

# Execute example scenario that will fail on /usr/bin/echo command
python ciot.py exec -T examples/echocat/tests -U examples/echocat/tests/unit-echo-wrong.yml -C /usr/bin/echo
# Scenario should fail

# Print out the parsed version of the unit-echo-wrong
python ciot.py parse -T examples/echocat/tests -U examples/echocat/tests/unit-echo-wrong.yml
# or you should be able to produce the whole structure as JSON
python ciot.py parse -T examples/echocat/tests -U examples/echocat/tests/unit-echo-wrong.yml -o json
```




