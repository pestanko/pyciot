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
from typing import List, Dict, Optional, Any, TypeVar, Type

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
    def as_dict(self) -> Dict:
        return dict_serialize(self, as_dict_skip=True)


GeneralDefType = TypeVar('GeneralDefType', bound='GeneralDef')


class GeneralDef(AsDict):
    """General definition - common base of all definitions"""

    def __init__(self, name: str, desc: str = None):
        self.name: str = name
        self.desc: str = desc if desc else name


class UnitDef(GeneralDef):
    def __init__(self, name: str, desc: str = None, tests: List['TestDef'] = None):
        super().__init__(name, desc)
        self.tests = tests or []


class TestDef(GeneralDef):
    def __init__(self, name: str, desc: str = None,
                 stdin: Path = None, exit_code: Optional[int] = None, args: List[str] = None,
                 env: Dict[str, Any] = None,
                 checks: List['CheckDef'] = None, unit: 'UnitDef' = None):
        super().__init__(name, desc)
        self.stdin: Optional[Path] = stdin
        self.args: List[str] = args
        self.exit_code: Optional[int] = exit_code
        self.checks: Optional[List['CheckDef']] = checks or []
        self.env: Dict[str, Any] = env
        self.unit = unit

    def as_dict(self) -> Dict:
        items = {k: v for k, v in self.__dict__.items() if k not in ('unit',)}
        return dict_serialize(items)


class CheckDef(GeneralDef):
    def __init__(self, name: str, desc: str = None, assertion: 'Assertion' = None):
        super().__init__(name, desc)
        self.assertion = assertion


class Assertion(AsDict):
    def __init__(self, kind: str, params: Dict[str, Any]):
        self.kind = kind
        self.params = params


##
# Parse
##


class UnitFileDefinitionParser:
    def __init__(self, test_dir: Path, data_dir: Path = None):
        self.test_dir = test_dir
        self.data_dir = data_dir

    def parse_unit(self, unit_file: Path) -> 'UnitDef':
        unit_data = load_file(unit_file)
        return self.parse_definition(unit_data, unit_file.stem)

    def parse_definition(self, df: Dict[str, Any], unit_name: str = None) -> UnitDef:
        unit_df = df.get('unit')
        if not unit_df:
            unit_df = {'name': unit_name, 'desc': unit_name}
        if 'name' not in df:
            unit_df['name'] = unit_name

        unit_definition = UnitDef(name=unit_df['name'], desc=unit_df.get('desc'))
        for test_df in df['tests']:
            parsed = self.parse_test_def(unit_definition, test_df)
            unit_definition.tests.extend(parsed)

        return unit_definition

    def parse_test_def(self, unit_definition: 'UnitDef', df: Dict[str, Any]) -> List['TestDef']:
        # general params
        name = df['name']
        desc = df.get('desc', name)

        if 'template' in df:
            return self.parse_test_template(unit_definition, df, name, desc)

        stdin = df.get('stdin')
        args = df.get('args', [])
        # Ability to explicitly set to None - null, if null, do not check
        exit_code = df['exit'] if 'exit' in df else 0
        checks = self.parse_checks(df)
        return [TestDef(name, desc, stdin, exit_code, args, checks=checks, unit=unit_definition)]

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

    def parse_checks(self, df: Dict[str, Any]) -> List['CheckDef']:
        checks = []
        stdout = df.get('out', df.get('stdout'))
        if stdout is not None:
            checks.append(self._file_assertion("@stdout", stdout))

        stderr = df.get('err', df.get('stderr'))
        if stderr is not None:
            checks.append(self._file_assertion("@stderr", stderr))

        exit_code = df.get('exit', df.get('exit_code', 0))
        if exit_code is not None:
            assertion = Assertion(ExitCodeAssertionRunner.NAME, dict(expected=exit_code))
            checks.append(CheckDef("exit_check", "Check the command exit code (main return value)",
                                   assertion))

        files = df.get('files')
        if files is not None and isinstance(files, dict):
            for prov, exp in files.items():
                check = self._file_assertion(prov, exp)
                checks.append(check)

        return checks

    def _file_assertion(self, selector: str, value):
        assertion = Assertion(
            FileAssertionRunner.NAME,
            dict(selector=selector, expected=self.data_dir / value)
        )
        check = CheckDef("file_check", f"Check the file content [{selector}]", assertion)
        return check


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
        if res.kind.is_fail():
            self.kind = res.kind


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


class DefinitionRunner:
    def __init__(self, paths: 'AppConfig'):
        self.paths = paths
        self.assertion_runners = AssertionRunners.instance()

    def run_unit(self, unit_df: UnitDef) -> 'UnitRunResult':
        LOG.info(f"[RUN] Running the unit: {unit_df.name}")
        unit_result = UnitRunResult(unit_df)
        unit_ws = self.paths.unit_workspace(unit_df.name)
        LOG.debug(f"[RUN] Creating unit workspace: {unit_ws}")
        for test_df in unit_df.tests:
            test_result = self.run_test(test_df, unit_ws)
            LOG.debug(f"[RUN] Test [{test_df.name}] result: {test_result.kind}")
            unit_result.add_subresult(test_result)
        LOG.debug(f"[RUN] Unit result: {unit_result.kind} ")
        return unit_result

    def run_test(self, test_df: 'TestDef', unit_ws: Path) -> 'TestRunResult':
        LOG.info(f"[RUN] Running the test{test_df.name} from {test_df.unit.name}")
        test_result = TestRunResult(test_df)

        try:
            cmd = str(self.paths.command)
            cmd_res = execute_cmd(cmd,
                                  args=test_df.args,
                                  stdin=test_df.stdin,
                                  nm=test_df.name,
                                  env=test_df.env,
                                  ws=unit_ws)
            test_result.cmd_result = cmd_res
            ctx = TestCtx(self.paths, test_df, cmd_res)
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
    nm = nm or cmd
    stdout = stdout or ws / f'{nm}.stdout'
    stderr = stderr or ws / f'{nm}.stderr'
    with stdout.open('w') as fd_out, stderr.open('w') as fd_err:
        fd_in = Path(stdin).open() if stdin else None
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
        return obj.as_dict()

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


def print_unit_df(df: 'UnitDef', colors: bool = True):
    tc = tcolors(colors)
    print(f"UNIT: [{tc.wrap(tc.GREEN, df.name)}]", f":: {df.desc}")
    for test in df.tests:
        print(
            f"- Test: [{tc.wrap(tc.CYAN, test.name)}] :: {test.desc} (Checks: {len(test.checks)})")
        for check in test.checks:
            print(
                f"\t * Check: [{tc.wrap(tc.MAGENTA, check.name)}] :: {check.desc} "
                f"[kind={check.assertion.kind}]"
            )


def print_unit_result(unit_res: 'UnitRunResult', with_checks: bool = False, colors: bool = True):
    tc = tcolors(colors)

    def _prk(r: 'GeneralResultType'):
        color = tc.RED if r.kind.is_fail() else tc.GREEN
        return tc.wrap(color, f"[{r.kind.value.upper()}]")

    def _p(r: 'GeneralResultType'):
        return f"{_prk(r)} ({r.df.name}) :: {r.df.desc}"

    print(_p(unit_res))
    for test_res in unit_res.tests:
        print(f"- {_p(test_res)}")
        if test_res.kind.is_fail():
            if test_res.message:
                print(f"\t Message: {test_res.message}")
        if test_res.kind.is_pass() and not with_checks:
            continue
        for ch_res in test_res.checks:
            if ch_res.kind.is_fail() or with_checks:
                print(f"\t* {_p(ch_res)}")

            if ch_res.kind.is_pass():
                continue

            print(ch_res.fail_msg("\t\t[info] "))

    print(f"\n\nOVERALL RESULT: {_prk(unit_res)}\n")


def dump_junit_report(unit_res: 'UnitRunResult', artifacts: Path) -> Path:
    try:
        import junitparser
    except ImportError:
        LOG.warning("No JUNIT generated - junit parser is not installed")
        return None
    report_path = artifacts / 'junit_report.xml'
    LOG.info(f"[REPORT] Generating JUNIT report: {report_path}")
    suites = junitparser.JUnitXml()
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


## App config

class AppConfig(AsDict):
    def __init__(self, command: str, tests_dir: Path, data_dir: Path = None,
                 artifacts: Path = None):
        tests_dir = Path(tests_dir) if tests_dir else Path.cwd()
        self.command: str = command
        self.tests_dir: Path = Path(tests_dir)
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
        sub.add_argument('-C', '--command', type=str,
                         help="Location of the command/binary you would like to test")
        sub.add_argument('-U', '--unit', type=str,
                         help='Location of the unit/test definition file')
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


def cli_parse(args):
    cfg = _get_app_cfg(args)
    unit_file = Path(args.unit)
    parser = UnitFileDefinitionParser(cfg.tests_dir, data_dir=cfg.data_dir)
    unit_df = parser.parse_unit(unit_file)
    if args.output in ['json', 'j']:
        print(dump_json(unit_df))
    else:
        print_unit_df(unit_df)
    return True


def _get_app_cfg(args):
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
    cfg = _get_app_cfg(args)
    unit_file = Path(args.unit)
    parser = UnitFileDefinitionParser(cfg.tests_dir, data_dir=cfg.data_dir)
    unit_df = parser.parse_unit(unit_file)
    runner = DefinitionRunner(cfg)
    result = runner.run_unit(unit_df)
    print_unit_result(result)
    ws = cfg.unit_workspace(unit_df.name)
    print(f"UNIT WORKSPACE: {ws}")
    report = dump_junit_report(unit_res=result, artifacts=ws)
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
