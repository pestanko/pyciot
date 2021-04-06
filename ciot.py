#! /usr/bin/env python3
import argparse
import copy
import enum
import json
import logging
import string
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, TypeVar, Type, Union

PYTHON_REQUIRED = "3.7"
APP_NAME = "ciot"
APP_VERSION = "0.0.1-alpha1"
APP_DESC = """
Command Input Output Testing Tool (ciot)

TBD
"""

LOG = logging.getLogger(APP_NAME)
GLOBAL_TIMEOUT = 10 * 60
FILE_EXTENSIONS = ('in', 'out', 'err', 'files', 'args')


##
# Definitions
##
class AsDict:
    AD_EXCLUDE = None

    def as_dict(self) -> Dict:
        exl = self.AD_EXCLUDE or []
        items = {k: v for k, v in self.__dict__.items() if k[0] != '_' or k not in exl}
        if self.AD_EXCLUDE:
            items = {k: v for k, v in items.items() if k not in self.AD_EXCLUDE}
        return dict_serialize(items, as_dict_skip=True)

    def as_dict_restrict(self) -> Dict:
        return self.as_dict()


GeneralDefType = TypeVar('GeneralDefType', bound='GeneralDef')


class GeneralDef(AsDict):
    """General definition - common base of all definitions"""

    def __init__(self, metadata: Dict[str, Any]):
        name = metadata['name']
        self.metadata = {
            'name': name,
            'desc': metadata.get('desc', name),
            'pts': metadata.get('pts', 0.0)
        }

    @property
    def name(self) -> str:
        return self.metadata['name']

    @property
    def desc(self) -> str:
        return self.metadata['desc']

    @property
    def pts(self) -> Optional[float]:
        return self.metadata['pts']


class SuitesDef(GeneralDef):
    def __init__(self, metadata: Dict[str, Any], units: List['UnitDef'] = None):
        super().__init__(metadata)
        self.units = units or []
        self.settings: Dict['str', Any] = {}
        self.overrides: List[Dict[str, Any]] = []


class UnitDef(GeneralDef):
    AD_EXCLUDE = ('units', 'suites')

    def __init__(self, metadata: Dict[str, Any],
                 tests: List['TestDef'] = None, suites: 'SuitesDef' = None):
        super().__init__(metadata)
        self.tests = tests or []
        self.suites = suites
        self.settings: Dict['str', Any] = {}


class TestDef(GeneralDef):
    AD_EXCLUDE = ('units', 'suites', 'unit')

    def __init__(self, metadata: Dict['str', Any],
                 stdin: Path = None, exit_code: Optional[int] = None, args: List[str] = None,
                 env: Dict[str, Any] = None,
                 checks: List['CheckDef'] = None, unit: 'UnitDef' = None):
        super().__init__(metadata)
        self.stdin: Optional[Path] = stdin
        self.args: List[str] = args
        self.exit_code: Optional[int] = exit_code
        self.checks: Optional[List['CheckDef']] = checks or []
        self.env: Dict[str, Any] = env
        self.unit = unit


class CheckDef(GeneralDef):
    AD_EXCLUDE = ('units', 'suites', 'unit', 'test')

    def __init__(self, name: str, desc: str = None, assertion: 'Assertion' = None,
                 test: Optional['TestDef'] = None):
        super().__init__({'name': name, 'desc': desc})
        self.assertion: 'Assertion' = assertion
        self.test: Optional['TestDef'] = test


class Assertion(AsDict):
    def __init__(self, kind: str, params: Dict[str, Any]):
        self.kind: str = kind
        self.params: Dict[str, Any] = params


##
# Parse
##


class DefinitionParser:
    def __init__(self, test_dir: Path, data_dir: Path = None):
        self.test_dir = test_dir
        self.data_dir = data_dir

    def parse(self, unit: Path = None, suites: Path = None) -> 'SuitesDef':
        if unit is not None:
            udf = self.parse_unit(unit)
            metadata = {'name': f"suites-{udf.name}", 'desc': udf.desc}
            sdf = SuitesDef(metadata, units=[udf])
            udf.suites = sdf
            return sdf

        return self.suites_parser(suites)

    def suites_parser(self, suites_file: Optional[Path]) -> SuitesDef:
        if suites_file:
            return self._parse_suites_file(suites_file)
        suites = SuitesDef(dict(name=self.test_dir.name))
        suites.units = self._find_units(suites)
        return suites

    def parse_unit(self, unit_file: Path, suites: 'SuitesDef' = None) -> Optional[UnitDef]:
        unit_data = load_file(unit_file)
        if 'tests' not in unit_data and 'unit' not in unit_data:
            # It is not a valid unit file
            return None
        LOG.debug(f"Found unit: {unit_file}")
        return self.parse_unit_definition(unit_data, unit_file.stem, suites=suites)

    def parse_suites_definition(self, df: Dict[str, Any], master_name: str = None) -> 'SuitesDef':
        metadata = self._parse_general_metadata(df, 'master', master_name)
        suites_def = SuitesDef(metadata)
        suites_def.settings = df.get('settings', {})
        suites_def.overrides = df.get('overrides', [])

        for unit_name in df['units']:
            units = self._find_units(suites_def, name=unit_name)
            suites_def.units.extend(units)
        return suites_def

    def parse_unit_definition(self, df: Dict[str, Any], unit_name: str = None,
                              suites: 'SuitesDef' = None) -> UnitDef:
        metadata = self._parse_general_metadata(df, 'unit', unit_name)
        unit_definition = UnitDef(metadata)
        unit_definition.suites = suites
        unit_definition.settings = df.get('settings', {})

        for test_df in df['tests']:
            parsed = self.parse_test_def(unit_definition, test_df)
            unit_definition.tests.extend(parsed)

        return unit_definition

    def parse_test_def(self, unit_definition: 'UnitDef', df: Dict[str, Any]) -> List['TestDef']:
        # general params
        name = df['name']
        desc = df.get('desc', name)
        pts = df.get('pts', 0.0)
        metadata = {
            'name': name,
            'desc': desc,
            'pts': pts,
        }

        if 'template' in df:
            return self.parse_test_template(unit_definition, df, name, desc)

        stdin = df.get('in', df.get('stdin'))
        if stdin:
            stdin = self._resolve_file(stdin)
        args = df.get('args', [])
        # Ability to explicitly set to None - null, if null, do not check
        exit_code = df['exit'] if 'exit' in df else 0
        test_df = TestDef(metadata, stdin, exit_code, args, checks=[], unit=unit_definition)
        test_df.checks = self.parse_checks(df, test_df)
        return [test_df]

    def parse_test_template(self, unit_definition: 'UnitDef', df: Dict[str, Any],
                            test_name: str, test_desc: str):
        template = df['template']
        tests = []
        for (idx, case) in enumerate(df['cases']):
            expanded = deep_template_expand(template, case['var'])
            cc = copy.deepcopy(case)
            del cc['var']
            case_df = {**expanded, **cc}
            case_name = case_df.get('name', idx)
            case_desc = case_df.get('desc', idx)
            case_df['name'] = f"{test_name}@{case_name}"
            case_df['desc'] = f"{test_desc} @ {case_desc}"
            tests.extend(self.parse_test_def(unit_definition, case_df))
        return tests

    def parse_checks(self, df: Dict[str, Any], test_df: 'TestDef') -> List['CheckDef']:
        checks = []
        stdout = df.get('out', df.get('stdout'))
        if stdout is not None:
            checks.append(self._file_assertion("@stdout", stdout, test_df))

        stderr = df.get('err', df.get('stderr'))
        if stderr is not None:
            checks.append(self._file_assertion("@stderr", stderr, test_df))

        exit_code = df.get('exit', df.get('exit_code', 0))
        if exit_code is not None:
            assertion = Assertion(ExitCodeAssertionRunner.NAME, dict(expected=exit_code))
            check = CheckDef("exit_check",
                             "Check the command exit code (main return value)",
                             assertion, test=test_df)
            checks.append(check)

        files = df.get('files')
        if files is not None and isinstance(files, dict):
            for prov, exp in files.items():
                check = self._file_assertion(prov, exp, test_df)
                checks.append(check)

        return checks

    def _parse_general_metadata(self, df: Dict[str, Any], metadata_name: str, default_name: str):
        general_df = df.get(metadata_name)
        if not general_df:
            general_df = {'name': default_name, 'desc': default_name, 'pts': None}
        if 'name' not in general_df:
            general_df['name'] = default_name
        return general_df

    def _file_assertion(self, selector: str, value, test_df: 'TestDef'):
        assertion = Assertion(
            FileAssertionRunner.NAME,
            dict(selector=selector, expected=self._resolve_file(value))
        )
        check = CheckDef("file_check", f"Check the file content [{selector}]",
                         assertion, test=test_df)
        return check

    def _resolve_file(self, value: Any) -> Union[Path, Dict]:
        if isinstance(value, Path) or isinstance(value, str):
            value = Path(value)
            return value if value.is_absolute() else (self.data_dir / value).resolve()
        if isinstance(value, dict):
            # TODO: implement file content provider
            return value
        return value

    def _find_units(self, suites: 'SuitesDef', name: str = '*') -> List['UnitDef']:
        units = []
        for unit_path in self.test_dir.glob(f"{name}.*"):
            if unit_path.suffix not in ['.json', '.yaml', '.yml']:
                continue
            unit = self.parse_unit(unit_file=unit_path, suites=suites)
            if unit:
                units.append(unit)
        return units

    def _parse_suites_file(self, suites_file: Path) -> Optional['SuitesDef']:
        suites_data = load_file(suites_file)
        if 'units' not in suites_data and 'master' not in suites_data:
            # It is not a valid unit file
            return None
        return self.parse_suites_definition(suites_data, suites_file.stem)


##
# Execute
##

class ResultKind(enum.Enum):
    PASS = "pass"
    FAIL = "fail"

    @classmethod
    def check(cls, predicate: bool) -> 'ResultKind':
        return ResultKind.PASS if predicate else ResultKind.FAIL

    def is_pass(self) -> bool:
        return self == self.PASS

    def is_fail(self) -> bool:
        return self == self.FAIL


GeneralResultType = TypeVar('GeneralResultType', bound='GeneralResult')


class GeneralResult(AsDict):
    @classmethod
    def mk_fail(cls, df: 'GeneralDefType', message: str) -> 'GeneralResultType':
        return cls(df, kind=ResultKind.FAIL, message=message)

    def __init__(self, df: 'GeneralDefType', kind: ResultKind = ResultKind.PASS,
                 message: str = None):
        self.df: 'GeneralDefType' = df
        self.kind = kind
        self.message: str = message
        self.detail: Optional[Dict[str, Any]] = None
        self.sub_results: List['GeneralResultType'] = []

    def add_subresult(self, res: 'GeneralResultType'):
        self.sub_results.append(res)
        if self.is_pass() and res.kind.is_fail():
            self.message = "Some of the sub-results has failed"
            self.kind = res.kind

    def is_pass(self) -> bool:
        return self.kind.is_pass()

    def is_fail(self) -> bool:
        return self.kind.is_fail()


class SuitesRunResult(GeneralResult):
    def __init__(self, df: 'SuitesDef'):
        super().__init__(df)

    @property
    def units(self) -> List['UnitRunResult']:
        return self.sub_results


class UnitRunResult(GeneralResult):
    def __init__(self, df: 'UnitDef'):
        super().__init__(df)

    @property
    def tests(self) -> List['TestRunResult']:
        return self.sub_results


class TestRunResult(GeneralResult):
    def __init__(self, df: 'TestDef'):
        super().__init__(df)
        self.cmd_result: Optional['CommandResult'] = None

    @property
    def checks(self) -> List['CheckResult']:
        return self.sub_results


class CheckResult(GeneralResult):
    def __init__(self, df: 'CheckDef', kind: ResultKind, message: str = "",
                 expected=None, provided=None, detail=None, diff=None):
        super().__init__(df, kind=kind, message=message)
        self.expected = expected
        self.provided = provided
        self.diff = diff
        self.detail: Optional[Dict[str, Any]] = detail

    def fail_msg(self, fill: str = ""):
        result = ""
        if self.message:
            result += f"{fill}Message: {self.message}\n"

        if self.expected is not None:
            result += f"{fill}Expected: {self.expected}\n"

        if self.provided is not None:
            result += f"{fill}Provided: {self.provided}\n"

        if self.diff is not None:
            result += f"{fill}Diff: {self.diff}\n"

        if self.detail:
            result += f"{fill}Detail: {self.detail}\n"
        return result


def _merge_settings(unit_df: 'UnitDef') -> Dict[str, Any]:
    suites = unit_df.suites
    settings = {**suites.settings, **unit_df.settings}
    if suites.overrides:
        for uo in suites.overrides:
            if uo['unit'] != unit_df.name:
                continue
            for k, v in uo.items():
                if k != 'unit':
                    settings[k] = v
    return settings


class DefinitionRunner:
    def __init__(self, paths: 'AppConfig'):
        self.paths = paths
        self.assertion_runners = AssertionRunners.instance()

    def run_suites(self, suites_df: 'SuitesDef') -> 'SuitesRunResult':
        LOG.info(f"[RUN] Running the suites: {suites_df.name}")
        suites_result = SuitesRunResult(suites_df)
        for unit in suites_df.units:
            unit_result = self.run_unit(unit)
            suites_result.add_subresult(unit_result)
        return suites_result

    def run_unit(self, unit_df: UnitDef) -> 'UnitRunResult':
        LOG.info(f"[RUN] Running the unit: {unit_df.name}")
        unit_result = UnitRunResult(unit_df)
        unit_ws = self.paths.unit_workspace(unit_df.name)
        settings = _merge_settings(unit_df)
        LOG.debug(f"[RUN] Creating unit workspace: {unit_ws}")
        for test_df in unit_df.tests:
            test_result = self.run_test(test_df, unit_ws, settings=settings)
            LOG.debug(f"[RUN] Test [{test_df.name}] result: {test_result.kind}")
            unit_result.add_subresult(test_result)
        LOG.debug(f"[RUN] Unit result: {unit_result.kind} ")
        return unit_result

    def run_test(self, test_df: 'TestDef', unit_ws: Path,
                 settings: Dict[str, Any] = None) -> 'TestRunResult':
        LOG.info(f"[RUN] Running the test{test_df.name} from {test_df.unit.name}")
        test_result = TestRunResult(test_df)
        cmd, args = self._get_command(test_df, unit_ws, settings)
        timeout = settings.get('timeout', GLOBAL_TIMEOUT)
        try:
            cmd_res = execute_cmd(cmd,
                                  args=args,
                                  stdin=test_df.stdin,
                                  nm=test_df.name,
                                  env=test_df.env,
                                  timeout=timeout,
                                  ws=unit_ws)
            test_result.cmd_result = cmd_res
            ctx = TestCtx(self.paths, test_df, cmd_res)
            if settings.get('valgrind', False):
                test_df.checks.append(CheckDef("valgrind", "Check the execution using valgrind",
                                               Assertion("")))
            for check_df in test_df.checks:
                check_result = self.run_check(ctx, check_df)
                LOG.debug(f"[RUN] Check {check_df.name} for"
                          f" test [{test_df.name}] result: {check_result.kind}")
                test_result.add_subresult(check_result)
            return test_result
        except Exception as e:
            LOG.error("Execution failed: ", e)
            test_result.kind = ResultKind.FAIL
            test_result.message = "Execution failed"
            return test_result

    def _get_command(self, test_df: 'TestDef', unit_ws: Path, settings):
        cmd = settings.get('command', self.paths.command)
        if not settings.get('valgrind', False):
            return cmd, test_df.args
        vg_cmd = 'valgrind'
        va_log_file = unit_ws / f"{test_df.name}.val"
        vg_args = [
            "--leak-check=full",
            "--track-fds=yes",
            "--child-silent-after-fork=yes",
            f'--log-file={va_log_file}'
        ]
        cmd_args = test_df.args
        return vg_cmd, [*vg_args, '--', cmd, *cmd_args]

    def run_check(self, ctx: 'TestCtx', check_df: 'CheckDef'):
        LOG.info(f"[RUN] Running Check: {check_df.name} for {ctx.test_df.name}")

        kind = check_df.assertion.kind
        assertion_runner = self.assertion_runners.get(kind)
        if assertion_runner is None:
            return CheckResult.mk_fail(check_df, f"Unable find assertion runner: {kind}")
        instance = assertion_runner(ctx, check_df)
        return instance.evaluate()


class TestCtx:
    def __init__(self, paths: 'AppConfig', test_df: 'TestDef', cmd_res: 'CommandResult'):
        self.paths = paths
        self.test_df = test_df
        self.cmd_res = cmd_res

    @property
    def ws(self) -> Path:
        return self.paths.unit_workspace(self.test_df.unit.name)


class AssertionRunner:
    NAME = None

    def __init__(self, ctx: 'TestCtx', check_df: 'CheckDef'):
        self.ctx: 'TestCtx' = ctx
        self.check_df: CheckDef = check_df

    @property
    def assertion(self) -> 'Assertion':
        return self.check_df.assertion

    @property
    def params(self) -> Dict['str', Any]:
        return self.assertion.params

    def evaluate(self) -> 'CheckResult':
        return CheckResult.mk_fail(self.check_df, "Unimplemented check")


class FileAssertionRunner(AssertionRunner):
    NAME = "file_cmp"

    def evaluate(self) -> 'CheckResult':
        expected = self.params['expected']
        selector = self.params['selector']
        provided = self._get_provided(selector)
        if not provided.exists():
            return CheckResult.mk_fail(self.check_df, f"Created file does not exists: {selector}")
        return self._compare_files(provided, expected)

    def _get_provided(self, selector: str):
        if selector == "@stdout":
            return self.ctx.cmd_res.stdout
        if selector == "@stderr":
            return self.ctx.cmd_res.stderr

        provided = Path(selector)
        if provided.is_absolute():
            return provided
        return (Path.cwd() / provided).resolve()

    def _compare_files(self, provided: Path, expected: Path):
        nm: str = expected.name

        diff_exec = execute_cmd(
            'diff',
            args=['-u', str(expected), str(provided)],
            ws=self.ctx.ws,
            nm=f"diff-{nm}"
        )
        return CheckResult(
            self.check_df,
            kind=ResultKind.check(diff_exec.exit == 0),
            message="Files content diff",
            expected=expected,
            provided=provided,
            diff=str(diff_exec.stdout),
            detail=diff_exec.as_dict(),
        )


class ExitCodeAssertionRunner(AssertionRunner):
    NAME = "exit_code"

    def evaluate(self) -> 'CheckResult':
        expected = self.params['expected']
        provided = self.ctx.cmd_res.exit
        return CheckResult(
            self.check_df,
            kind=ResultKind.check(provided == expected),
            message="Exit code status",
            provided=provided,
            expected=expected,
            diff="provided != expected"
        )


class ValgrindAssertionRunner(AssertionRunner):
    NAME = "valgrind"

    def evaluate(self) -> 'CheckResult':
        report = self.ctx.ws / f"{self.ctx.test_df.name}.val"
        if not report.exists():
            return CheckResult.mk_fail(self.check_df, "Valgrind report does not exists!")
        number_of_errors = self._process_report(report)
        return CheckResult(
            self.check_df,
            kind=ResultKind.check(number_of_errors == 0),
            message="Valgrind validation failed",
            provided=number_of_errors,
            expected="0",
            diff=f"{number_of_errors} != 0",
            detail={
                'report': report,
            }
        )

    def _process_report(self, report: Path) -> int:
        # TODO
        return 0


class AssertionRunners:
    INSTANCE = None

    @classmethod
    def instance(cls) -> 'AssertionRunners':
        if cls.INSTANCE is None:
            cls.INSTANCE = cls.make()
        return cls.INSTANCE

    @classmethod
    def make(cls) -> 'AssertionRunners':
        instance = AssertionRunners()
        instance.add(FileAssertionRunner.NAME, FileAssertionRunner)
        instance.add(ExitCodeAssertionRunner.NAME, ExitCodeAssertionRunner)
        return instance

    def __init__(self):
        self.register: Dict[str, Type[AssertionRunner]] = {}

    def add(self, kind: str, runner: Type[AssertionRunner]):
        self.register[kind] = runner

    def get(self, kind: 'str') -> Optional[Type[AssertionRunner]]:
        return self.register.get(kind)


##
# Utils
##

def load_file(file: Path) -> Any:
    ext = file.suffix
    if ext == '.json':
        with file.open('r') as fd:
            return json.load(fd)
    if ext in ['.yml', '.yaml']:
        try:
            import yaml
        except Exception as ex:
            LOG.error("PyYaml library is not installed")
            raise ex
        with file.open('r') as fd:
            return yaml.safe_load(fd)

    raise Exception(f"Unsupported format: {ext} for {file}")


def deep_template_expand(template: Any, variables: Dict[str, Any]):
    if template is None:
        return None
    if isinstance(template, str):
        return string.Template(template).safe_substitute(variables)
    if isinstance(template, list):
        return [deep_template_expand(i, variables) for i in template]
    if isinstance(template, dict):
        return {k: deep_template_expand(v, variables) for k, v in template.items()}
    return template


def execute_cmd(cmd: str, args: List[str], ws: Path, stdin: Optional[Path] = None,
                stdout: Path = None, stderr: Path = None, nm: str = None,
                log: logging.Logger = None, **kwargs) -> 'CommandResult':
    log = log or LOG
    log.info(f"[CMD]: {cmd} with args {args}")
    log.debug(f"[CMD]: {cmd} with stdin {stdin}")
    nm = nm or cmd
    stdout = stdout or ws / f'{nm}.stdout'
    stderr = stderr or ws / f'{nm}.stderr'
    with stdout.open('w') as fd_out, stderr.open('w') as fd_err:
        fd_in = Path(stdin).open('r') if stdin else None
        start_time = time.perf_counter_ns()
        exec_result = subprocess.run(
            [cmd, *args],
            stdout=fd_out,
            stderr=fd_err,
            stdin=fd_in,
            **kwargs
        )
        if fd_in:
            fd_in.close()
    end_time = time.perf_counter_ns()
    log.info(f"[CMD] Result: {exec_result}")
    log.debug(f" -> Command stdout {stdout}")
    log.debug(f"STDOUT: {stdout.read_bytes()}")
    log.debug(f" -> Command stderr {stderr}")
    log.debug(f"STDERR: {stderr.read_bytes()}")

    return CommandResult(
        exit_code=exec_result.returncode,
        elapsed=end_time - start_time,
        stdout=stdout,
        stderr=stderr,
    )


class CommandResult(AsDict):
    def __init__(self, exit_code: int, stdout: Path, stderr: Path, elapsed: int):
        self.exit = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed = elapsed

    def as_dict(self) -> Dict:
        return {
            'exit': self.exit,
            'stdout': str(self.stdout),
            'stderr': str(self.stderr),
            'elapsed': self.elapsed,
        }


def dict_serialize(obj, as_dict_skip: bool = False) -> Any:
    if obj is None or isinstance(obj, str) or isinstance(obj, int):
        return obj
    if isinstance(obj, list):
        return [dict_serialize(i) for i in obj]

    if isinstance(obj, set):
        return {dict_serialize(i) for i in obj}

    if isinstance(obj, dict):
        return {k: dict_serialize(v) for k, v in obj.items()}

    if isinstance(obj, enum.Enum):
        return obj.value

    if not as_dict_skip and isinstance(obj, AsDict):
        return obj.as_dict_restrict()

    if hasattr(obj, '__dict__'):
        return {k: dict_serialize(v) for k, v in obj.__dict__.items()}

    if isinstance(obj, Path):
        return str(obj)

    return str(obj)


##
# Main CLI
##

# Printers

COLORS = ('black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white')


def _clr_index(name: str) -> int:
    try:
        return COLORS.index(name.lower())
    except Exception:
        return 7


def _clr(name: str, bright: bool = False):
    prefix = '\033['
    name = name.lower()
    if name == 'end':
        return f'{prefix}0m'
    if name == 'bold':
        return f'{prefix}1m'
    if name == 'underline':
        return f'{prefix}4m'
    mode = '9' if bright else '3'
    return f'{prefix}{mode}{_clr_index(name)}m'


class tcolors:
    BLUE = '\033[34m'
    CYAN = _clr('cyan')
    GREEN = _clr('green')
    MAGENTA = _clr('magenta')
    YELLOW = _clr('yellow')
    RED = _clr('red')
    ENDC = _clr('end')
    BOLD = _clr('bold')
    UNDERLINE = _clr('underline')

    def __init__(self, colors: bool = True):
        self._colors = colors

    def fail(self, s: str) -> str:
        return self.wrap(self.RED, s)

    def passed(self, s: str) -> str:
        return self.wrap(self.GREEN, s)

    def warn(self, s: str) -> str:
        return self.wrap(self.YELLOW, s)

    def head(self, s: str) -> str:
        return self.wrap(self.MAGENTA, s)

    def wrap(self, color_prefix: str, s: str) -> str:
        if not self._colors:
            return s
        return f"{color_prefix}{s}{self.ENDC}"


def print_suites_df(sdf: 'SuitesDef', colors: bool = True):
    tc = tcolors(colors)
    print(f"SUITES: [{tc.wrap(tc.GREEN, sdf.name)}] :: {sdf.desc}")
    for df in sdf.units:
        print(f"\tUNIT: [{tc.wrap(tc.GREEN, df.name)}]", f":: {df.desc}")
        for test in df.tests:
            print(
                f"\t- Test: [{tc.wrap(tc.CYAN, test.name)}] :: {test.desc} (Checks: {len(test.checks)})")
            for check in test.checks:
                print(
                    f"\t\t * Check: [{tc.wrap(tc.MAGENTA, check.name)}] :: {check.desc} "
                    f"[kind={check.assertion.kind}]"
                )
        print()


def print_suites_result(suites_res: 'SuitesRunResult', with_checks: bool = False,
                        colors: bool = True):
    tc = tcolors(colors)

    def _prk(r: 'GeneralResultType'):
        color = tc.RED if r.kind.is_fail() else tc.GREEN
        return tc.wrap(color, f"[{r.kind.value.upper()}]")

    def _p(r: 'GeneralResultType', t: str):
        return f"{_prk(r)} {t.capitalize()}: ({r.df.name}) :: {r.df.desc}"

    print(_p(suites_res, 'Suite'))
    for unit_res in suites_res.units:
        print(">>>", _p(unit_res, 'Unit'))
        for test_res in unit_res.tests:
            print(f"\t - {_p(test_res, 'Test')}")
            if test_res.kind.is_fail():
                if test_res.message:
                    print(f"\t\t Message: {test_res.message}")
            if test_res.kind.is_pass() and not with_checks:
                continue
            for ch_res in test_res.checks:
                if ch_res.kind.is_fail() or with_checks:
                    print(f"\t\t* {_p(ch_res, 'Check')}")

                if ch_res.kind.is_pass():
                    continue

                print(ch_res.fail_msg("\t\t  [info] "))
        print()

    print(f"\nOVERALL RESULT: {_prk(suites_res)}\n")


def dump_junit_report(suites_res: 'SuitesRunResult', artifacts: Path) -> Optional[Path]:
    try:
        import junitparser
    except ImportError:
        LOG.warning("No JUNIT generated - junit parser is not installed")
        return None
    report_path = artifacts / 'junit_report.xml'
    LOG.info(f"[REPORT] Generating JUNIT report: {report_path}")
    suites = junitparser.JUnitXml(suites_res.df.name)
    for unit_res in suites_res.units:
        unit_suite = junitparser.TestSuite(name=unit_res.df.name)
        for test_res in unit_res.tests:
            junit_case = junitparser.TestCase(
                name=test_res.df.desc,
                classname=test_res.df.unit.name + '/' + test_res.df.name,
                time=test_res.cmd_result.elapsed / 1000000.0 if test_res.cmd_result else 0
            )
            if test_res.kind.is_pass():
                continue
            fails = []
            for c in test_res.checks:
                fail = junitparser.Failure(c.message)
                fail.text = "\n" + c.fail_msg()
                fails.append(fail)
            junit_case.result = fails
            if test_res.cmd_result:
                junit_case.system_out = str(test_res.cmd_result.stdout)
                junit_case.system_err = str(test_res.cmd_result.stderr)
            unit_suite.add_testcase(junit_case)
        suites.add_testsuite(unit_suite)

    suites.write(str(report_path))
    return report_path


def _resolve_def_file(name: str, root: Path) -> Optional['Path']:
    if '.' in name:
        pth = Path(name)
        return pth if pth.exists() else root / pth
    for ext in ('json', 'yml', 'yaml'):
        pth = root / f"{name}.{ext}"
        if pth.exists():
            return pth
    return None


## App config

class AppConfig(AsDict):
    def __init__(self, command: str, tests_dir: Path, data_dir: Path = None,
                 artifacts: Path = None):
        tests_dir = Path(tests_dir) if tests_dir else Path.cwd()
        self.command: str = command
        self.tests_dir: Path = Path(tests_dir).resolve()
        self.data_dir: Path = Path(data_dir) if data_dir else _resolve_data_dir(tests_dir)
        self.artifacts: Path = Path(artifacts) if artifacts else _make_artifacts_dir()

    def unit_workspace(self, name: str) -> Path:
        ws = self.artifacts / name
        if not ws.exists():
            ws.mkdir(parents=True)
        return ws


def _resolve_data_dir(test_dir: Path) -> Path:
    if (test_dir / 'data').exists():
        return test_dir / 'data'
    return test_dir


def _make_artifacts_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix=APP_NAME + "-"))


def make_cli_parser() -> argparse.ArgumentParser:
    def _locations(sub):
        sub.add_argument('-C', '--command', type=str, default=None,
                         help="Location of the command/binary you would like to test")
        sub.add_argument('-U', '--unit', type=str,
                         help='Location or name of the unit/test definition file')
        sub.add_argument('-S', '--suites', type=str,
                         help='Location or name of the suites/master definition file')
        sub.add_argument('-T', '--test-files', type=str, help='Location of the test files',
                         default='tests')
        sub.add_argument('-D', '--test-data-files', type=str,
                         help='Location of the test data files',
                         default=None)
        sub.add_argument('-A', '--artifacts', type=str,
                         help='Location of the testing outputs/artifacts',
                         default=None)

    parser = argparse.ArgumentParser(APP_NAME, description=APP_DESC)
    parser.set_defaults(func=None)
    parser.add_argument("-L", "--log-level", type=str,
                        help="Set log level (DEBUG|INFO|WARNING|ERROR)", default='ERROR')
    subs = parser.add_subparsers(title="Sub-Commands")
    # Parse
    sub_parse = subs.add_parser("parse", help="Parse and print the mini hw scenario")
    sub_parse.add_argument("-o", "--output", help="Output format (console|json)", default="console")
    _locations(sub_parse)
    sub_parse.set_defaults(func=cli_parse)

    # Exec
    sub_exec = subs.add_parser("exec", help="Execute the unit file")
    _locations(sub_exec)
    sub_exec.set_defaults(func=cli_exec)
    return parser


def _resolve_definition(cfg: AppConfig, args):
    if args.unit:
        return {'unit': _resolve_def_file(args.unit, cfg.tests_dir)}
    if args.suites:
        return {'suites': _resolve_def_file(args.suites, cfg.tests_dir)}
    return {}


def cli_parse(args):
    cfg = _app_get_cfg(args)
    suites_df = _app_parse_suites(cfg, args)
    if args.output in ['json', 'j']:
        print(dump_json(suites_df))
    else:
        print_suites_df(suites_df)
    return True


def _app_parse_suites(cfg, args):
    defn = _resolve_definition(cfg, args)
    parser = DefinitionParser(cfg.tests_dir, data_dir=cfg.data_dir)
    suites_df = parser.parse(**defn)
    return suites_df


def _app_get_cfg(args):
    tests_dir = args.test_files
    data_dir = args.test_data_files
    artifacts = args.artifacts
    app_cfg = AppConfig(
        tests_dir=tests_dir,
        data_dir=data_dir,
        command=args.command,
        artifacts=artifacts
    )
    LOG.debug(f"[PATHS] Binary: {app_cfg.command}")
    LOG.debug(f"[PATHS] Test dir: {app_cfg.tests_dir}")
    LOG.debug(f"[PATHS] Test data dir: {app_cfg.data_dir}")
    LOG.debug(f"[PATHS] Artifacts: {app_cfg.artifacts}")
    return app_cfg


def cli_exec(args):
    cfg = _app_get_cfg(args)
    suites_df = _app_parse_suites(cfg, args)
    runner = DefinitionRunner(cfg)
    result = runner.run_suites(suites_df)
    print_suites_result(result)
    ws = cfg.unit_workspace(suites_df.name)
    print(f"UNIT WORKSPACE: {ws}")
    report = dump_junit_report(result, artifacts=ws)
    if report:
        print(f"JUNIT REPORT: {report}")

    return result.kind.is_pass()


def main():
    parser = make_cli_parser()
    args = parser.parse_args()
    _load_logger(args.log_level)
    LOG.debug(f"Parsed args: {args} ")
    if not args.func:
        parser.print_help()
        return
    if not args.func(args):
        print("\nExecution failed!")


def dump_json(obj, indent=4):
    def _dumper(x):
        if isinstance(x, AsDict):
            return x.as_dict()
        if isinstance(x, Path):
            return str(x)
        return x.__dict__

    return json.dumps(obj, default=_dumper, indent=indent)


def _load_logger(level: str = 'INFO'):
    level = level.upper()
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': '%(levelname)s %(asctime)s %(module)s %(message)s'
            },
            'simple': {
                'format': '%(levelname)s %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'simple'
            },
        },
        'loggers': {
            APP_NAME: {
                'handlers': ['console'],
                'level': level,
            }
        }
    }
    import logging.config
    logging.config.dictConfig(log_config)


if __name__ == '__main__':
    main()
