#! /usr/bin/env python3
import argparse
import copy
import enum
import json
import logging
import os
import shutil
import string
import subprocess
import tempfile
import time
from dataclasses import dataclass
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
GLOBAL_TIMEOUT = 5
FILE_EXTENSIONS = ('in', 'out', 'err', 'files', 'args')

DParams = Dict[str, Any]


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


@dataclass(frozen=True)
class DefinitionMetadata:
    name: str
    desc: str
    pts: float = 0.0

    @classmethod
    def from_dict(cls, c: DParams) -> 'DefinitionMetadata':
        return DefinitionMetadata(
            name=c['name'],
            desc=c.get('desc', c['name']),
            pts=c.get('pts', 0.0)
        )


class GeneralDef(AsDict):
    """General definition - common base of all definitions"""

    def __init__(self, metadata: DefinitionMetadata):
        self.metadata: DefinitionMetadata = metadata

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def desc(self) -> str:
        return self.metadata.desc

    @property
    def pts(self) -> Optional[float]:
        return self.metadata.pts


class SuiteDef(GeneralDef):
    def __init__(self, metadata: DefinitionMetadata, units: List['UnitDef'] = None):
        super().__init__(metadata)
        self.units = units or []
        self.settings: Dict['str', Any] = {}
        self.overrides: List[Dict[str, Any]] = []


class UnitDef(GeneralDef):
    AD_EXCLUDE = ('units', 'suite')

    def __init__(self, metadata: DefinitionMetadata,
                 tests: List['TestDef'] = None, suite: 'SuiteDef' = None):
        super().__init__(metadata)
        self.pre_tasks: List['TaskDef'] = []
        self.tests: List['TestDef'] = tests or []
        self.suite = suite
        self.settings: Dict['str', Any] = {}


class TestDef(GeneralDef):
    AD_EXCLUDE = ('units', 'suite', 'unit')

    def __init__(self, metadata: DefinitionMetadata,
                 stdin: Union[str, Dict] = None,
                 exit_code: Optional[int] = None, args: List[str] = None,
                 env: Dict[str, Any] = None,
                 checks: List['TaskDef'] = None, unit: 'UnitDef' = None):
        super().__init__(metadata)
        self.pre_tasks: List['TaskDef'] = []
        self.stdin: Union[str, Dict] = stdin
        self.args: List[str] = args
        self.exit_code: Optional[int] = exit_code
        self.checks: Optional[List['TaskDef']] = checks or []
        self.env: Dict[str, Any] = env
        self.unit = unit


class TaskDef(GeneralDef):
    AD_EXCLUDE = ('units', 'suite', 'unit', 'test')

    def __init__(self, name: str, desc: str = None, kind: str = None,
                 params: DParams = None, test: Optional['TestDef'] = None):
        super().__init__(DefinitionMetadata(name, desc))
        self.kind: str = kind or self.name
        self.params: 'DParams' = params
        self.test: Optional['TestDef'] = test


##
# Parse
##


def _resolve_file(value: Any, root: Path) -> Union[Path, Dict, None]:
    if isinstance(value, Path) or isinstance(value, str):
        value = Path(value)
        return value if value.is_absolute() else (root / value).resolve()
    if isinstance(value, dict):
        # TODO: implement file content provider
        return value
    return value


class DefinitionParser:
    def __init__(self, test_dir: Path, data_dir: Path = None):
        self.test_dir = test_dir
        self.data_dir = data_dir

    def parse(self, unit: Path = None, suite: Path = None) -> 'SuiteDef':
        if unit is not None:
            udf = self.parse_unit(unit)
            metadata = DefinitionMetadata(f"suite-{udf.name}", udf.desc)
            sdf = SuiteDef(metadata, units=[udf])
            udf.suite = sdf
            return sdf

        return self.suite_parser(suite)

    def suite_parser(self, suite_file: Optional[Path]) -> SuiteDef:
        if suite_file:
            return self._parse_suite_file(suite_file)
        metadata = DefinitionMetadata.from_dict({'name': self.test_dir.name})
        suite = SuiteDef(metadata)
        suite.units = self._find_units(suite)
        return suite

    def parse_unit(self, unit_file: Path, suite: 'SuiteDef' = None) -> Optional[UnitDef]:
        unit_data = load_file(unit_file)
        if 'tests' not in unit_data and 'unit' not in unit_data:
            # It is not a valid unit file
            return None
        if 'skip_on' in unit_data and APP_NAME in unit_data['skip_on']:
            LOG.warning(f"[PARSE] Skipping unit: {unit_file}")
            return None
        LOG.debug(f"Found unit: {unit_file}")
        return self.parse_unit_definition(unit_data, unit_file.stem, suite=suite)

    def parse_suite_definition(self, df: Dict[str, Any], master_name: str = None) -> 'SuiteDef':
        metadata = self._parse_metadata(df, 'master', master_name)
        suite_def = SuiteDef(metadata)
        suite_def.settings = df.get('settings', {})
        suite_def.overrides = df.get('overrides', [])

        for unit_name in df['units']:
            units = self._find_units(suite_def, name=unit_name)
            suite_def.units.extend(units)
        return suite_def

    def parse_unit_definition(self, df: Dict[str, Any], unit_name: str = None,
                              suite: 'SuiteDef' = None) -> UnitDef:
        metadata = self._parse_metadata(df, 'unit', unit_name)
        unit_definition = UnitDef(metadata)
        unit_definition.suite = suite
        unit_definition.settings = df.get('settings', {})

        for idx, test_df in enumerate(df['tests']):
            parsed = self.parse_test_def(unit_definition, test_df, idx=idx)
            unit_definition.tests.extend(parsed)

        return unit_definition

    def parse_test_def(self, unit_definition: 'UnitDef',
                       df: Dict[str, Any], idx: int = 0) -> List['TestDef']:
        # general params
        metadata = self._parse_metadata(df, metadata_name=None, default_name=f'test-{idx}')

        if 'template' in df:
            return self.parse_test_template(unit_definition, df, metadata)

        stdin = df.get('in', df.get('stdin'))
        args = df.get('args', [])
        # Ability to explicitly set to None - null, if null, do not check
        exit_code = df['exit'] if 'exit' in df else 0
        test_df = TestDef(metadata, stdin, exit_code, args, checks=[], unit=unit_definition)
        test_df.checks = self._parse_checks(df, test_df)
        test_df.pre_tasks = self._parse_pre_tasks(df, test_df)
        return [test_df]

    def parse_test_template(self, unit_definition: 'UnitDef', df: Dict[str, Any],
                            test_metadata: DefinitionMetadata):
        template = df['template']
        tests = []
        for (idx, case) in enumerate(df['cases']):
            expanded = deep_template_expand(template, case['var'])
            cc = copy.deepcopy(case)
            del cc['var']
            case_df = {**expanded, **cc}
            case_name = case_df.get('name', idx)
            case_desc = case_df.get('desc', idx)
            case_df['name'] = f"{test_metadata.name}@{case_name}"
            case_df['desc'] = f"{test_metadata.desc} @ {case_desc}"
            tests.extend(self.parse_test_def(unit_definition, case_df))
        return tests

    def _parse_checks(self, df: Dict[str, Any], test_df: 'TestDef') -> List['TaskDef']:
        checks = []
        stdout = df.get('out', df.get('stdout'))
        if not _should_ignore(stdout):
            checks.append(self._file_check("@stdout", stdout, test_df))

        stderr = df.get('err', df.get('stderr'))
        if not _should_ignore(stderr):
            checks.append(self._file_check("@stderr", stderr, test_df))

        exit_code = df.get('exit', df.get('exit_code', 0))
        if not _should_ignore(exit_code):
            check = TaskDef(ExitCodeCheckTask.NAME,
                            desc="Check the command exit code (main return value)",
                            params=dict(expected=exit_code), test=test_df)
            checks.append(check)

        files = df.get('files')
        if files is not None and isinstance(files, dict):
            for prov, exp in files.items():
                check = self._file_check(prov, exp, test_df)
                checks.append(check)

        return checks

    def _parse_pre_tasks(self, df: Dict[str, Any], test_df: 'TestDef'):
        tasks = []
        data = df.get('data')
        if data:
            tasks.append(
                TaskDef(CopyFilesTask.NAME, kind=CopyFilesTask.NAME, params={'files': data})
            )
        return tasks

    def _parse_metadata(self, df: Dict[str, Any], metadata_name: Optional[str],
                        default_name: str) -> DefinitionMetadata:
        general_df = df.get(metadata_name) if metadata_name else df
        if not general_df and metadata_name:
            general_df = {'name': default_name, 'desc': default_name, 'pts': 0.0}
        if 'name' not in general_df:
            general_df['name'] = default_name
        return DefinitionMetadata.from_dict(general_df)

    def _file_check(self, selector: str, value, test_df: 'TestDef'):
        return TaskDef(FileCheckTask.NAME,
                       desc=f"Check the file content [{selector}]",
                       params=dict(selector=selector, expected=value),
                       test=test_df)

    def _find_units(self, suite: 'SuiteDef', name: str = '*') -> List['UnitDef']:
        units = []
        for unit_path in self.test_dir.glob(f"*{name}.*"):
            if unit_path.suffix not in ['.json', '.yaml', '.yml']:
                continue
            unit = self.parse_unit(unit_file=unit_path, suite=suite)
            if unit:
                units.append(unit)
        return units

    def _parse_suite_file(self, suite_file: Path) -> Optional['SuiteDef']:
        suite_data = load_file(suite_file)
        if 'units' not in suite_data and 'master' not in suite_data:
            # It is not a valid unit file
            return None
        return self.parse_suite_definition(suite_data, suite_file.stem)


def _should_ignore(value) -> bool:
    return value is None or value in ['ignore', 'any']


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

    @classmethod
    def mk_pass(cls, df: 'GeneralDefType') -> 'GeneralResultType':
        return cls(df, kind=ResultKind.PASS, message="Passed!")

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


class SuiteRunResult(GeneralResult):
    def __init__(self, df: 'SuiteDef'):
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
    def __init__(self, df: 'TaskDef', kind: ResultKind, message: str = "",
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
    suite = unit_df.suite
    settings = {**suite.settings, **unit_df.settings}
    if suite.overrides:
        for uo in suite.overrides:
            if uo['unit'] != unit_df.name:
                continue
            for k, v in uo.items():
                if k != 'unit':
                    settings[k] = v
    return settings


class DefinitionRunner:
    def __init__(self, paths: 'AppConfig'):
        self.paths = paths
        self.task_runners = TaskRunners.instance()

    def run_suite(self, suite_df: 'SuiteDef') -> 'SuiteRunResult':
        LOG.info(f"[RUN] Running the suite: {suite_df.name}")
        suite_result = SuiteRunResult(suite_df)
        for unit in suite_df.units:
            unit_result = self.run_unit(unit)
            suite_result.add_subresult(unit_result)
        return suite_result

    def run_unit(self, unit_df: UnitDef) -> 'UnitRunResult':
        LOG.info(f"[RUN] Running the unit: {unit_df.name}")
        unit_result = UnitRunResult(unit_df)
        unit_ws = self.paths.unit_workspace(unit_df.suite.name, unit_df.name)
        settings = _merge_settings(unit_df)
        LOG.debug(f"[RUN] Creating unit workspace: {unit_ws}")

        for task_df in unit_df.pre_tasks:
            task_result = self.run_task(task_df, unit_ws, settings=settings)
            LOG.debug(f"[RUN] Task [{task_df.kind}] result: {task_result.kind}")
            unit_result.add_subresult(task_result)
            if task_result.is_fail():
                LOG.warning("[RUN] Unit execution ended since one of the pre-tasks has failed")
                return unit_result

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
        ctx = TestCtx(self.paths, test_df, settings)

        try:
            ctx.cmd_res = self._execute_tested_command(ctx)
        except Exception as e:
            LOG.error("Execution failed: ", e)
            test_result.kind = ResultKind.FAIL
            test_result.message = f"Execution failed: {e}"
            return test_result

        for check_df in test_df.checks:
            check_result = self.run_check(ctx, check_df)
            LOG.debug(f"[RUN] Check {check_df.name} for"
                      f" test [{test_df.name}] result: {check_result.kind}")
            test_result.add_subresult(check_result)
        return test_result

    def _execute_tested_command(self, ctx: 'TestCtx') -> 'CommandResult':
        cmd, args = self._get_command(ctx.test_df, ctx.ws, ctx.settings)
        timeout = ctx.settings.get('timeout', GLOBAL_TIMEOUT)
        data_dir = Path(ctx.settings.get('data', self.paths.data_dir))
        stdin = _resolve_file(ctx.test_df.stdin, root=data_dir)

        return execute_cmd(cmd,
                           args=args,
                           stdin=stdin,
                           nm=ctx.test_df.name,
                           env=ctx.test_df.env,
                           timeout=timeout,
                           ws=ctx.ws)

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

    def run_check(self, ctx: 'TestCtx', task_df: 'TaskDef'):
        LOG.info(f"[RUN] Running Check: {task_df.name} for {ctx.test_df.name}")

        kind = task_df.kind
        task_runner = self.task_runners.get(kind)
        if task_runner is None:
            return CheckResult.mk_fail(task_df, f"Unable find config runner: {kind}")
        instance = task_runner(ctx, task_df)
        return instance.evaluate()

    def run_task(self, task_df, unit_ws, settings) -> 'CheckResult':
        return None


class TestCtx:
    def __init__(self, paths: 'AppConfig', test_df: 'TestDef',
                 settings: Dict[str, Any], cmd_res: 'CommandResult' = None):
        self.paths = paths
        self.test_df = test_df
        self.settings = settings
        self.cmd_res = cmd_res

    @property
    def ws(self) -> Path:
        return self.paths.unit_workspace(self.test_df.unit.suite.name, self.test_df.unit.name)

    @property
    def data_dir(self) -> Path:
        return Path(self.settings.get('data', self.paths.data_dir))

    def resolve_dir(self, file_type: str) -> Path:
        mapping = {
            'data': self.data_dir,
            'tests': self.paths.tests_dir,
            'artifacts': self.ws,
        }
        return mapping.get(file_type.lower(), self.ws) if file_type else self.ws


class TaskRunner:
    NAME = None

    def __init__(self, ctx: 'TestCtx', check_df: 'TaskDef'):
        self.ctx: 'TestCtx' = ctx
        self.check_df: TaskDef = check_df

    @property
    def task_df(self) -> 'TaskDef':
        return self.check_df

    @property
    def params(self) -> Dict['str', Any]:
        return self.task_df.params

    def evaluate(self) -> 'CheckResult':
        return CheckResult.mk_fail(self.check_df, "Unimplemented check")


class FileCheckTask(TaskRunner):
    NAME = "file_cmp"

    def evaluate(self) -> 'CheckResult':
        expected = _resolve_file(self.params['expected'], self.ctx.data_dir)
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


class ExistFilesTasks(TaskRunner):
    NAME = "exist_file"

    def evaluate(self) -> 'CheckResult':
        files = self.params['files']
        for fdf in files:
            file = fdf['file']
            fdir: Path = self.ctx.resolve_dir(fdf.get('type', 'artifacts'))
            pth = fdir / file
            if not pth.exists():
                return CheckResult(
                    self.check_df, ResultKind.FAIL,
                    message="Required file does not exists",
                    provided=pth,
                    expected="exists",
                    diff=f"File does not exists: {pth}"
                )
        return CheckResult.mk_pass(self.check_df)


class CopyFilesTask(TaskRunner):
    NAME = "copy_files"

    def evaluate(self) -> 'CheckResult':
        files = self.params['files']
        for fpattern in files:
            for fp in self.ctx.data_dir.glob(fpattern):
                dest = self.ctx.ws / fp.name
                LOG.debug(f"[COPY] Copying file: {fp} to {dest}")
                shutil.copy2(fp, dest)

        return CheckResult.mk_pass(self.check_df)


class ExitCodeCheckTask(TaskRunner):
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


class ValgrindTask(TaskRunner):
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


class TaskRunners:
    INSTANCE = None

    @classmethod
    def instance(cls) -> 'TaskRunners':
        if cls.INSTANCE is None:
            cls.INSTANCE = cls.make()
        return cls.INSTANCE

    @classmethod
    def make(cls) -> 'TaskRunners':
        instance = TaskRunners()
        instance.add(FileCheckTask.NAME, FileCheckTask)
        instance.add(ExitCodeCheckTask.NAME, ExitCodeCheckTask)
        return instance

    def __init__(self):
        self.register: Dict[str, Type[TaskRunner]] = {}

    def add(self, kind: str, runner: Type[TaskRunner]):
        self.register[kind] = runner

    def get(self, kind: 'str') -> Optional[Type[TaskRunner]]:
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


def _norm_template(template: str) -> str:
    return template.replace("%<", "${").replace(">", "}")


def deep_template_expand(template: Any, variables: Dict[str, Any]):
    if template is None:
        return None
    if isinstance(template, str):
        normalized = _norm_template(template)
        return string.Template(normalized).safe_substitute(variables)
    if isinstance(template, list):
        return [deep_template_expand(i, variables) for i in template]
    if isinstance(template, dict):
        return {k: deep_template_expand(v, variables) for k, v in template.items()}
    return template


def execute_cmd(cmd: str, args: List[str], ws: Path, stdin: Optional[Path] = None,
                stdout: Path = None, stderr: Path = None, nm: str = None,
                log: logging.Logger = None, timeout: int = GLOBAL_TIMEOUT,
                env: Dict[str, Any] = None, cwd: Union[str, Path] = None,
                **kwargs) -> 'CommandResult':
    log = log or LOG
    log.info(f"[CMD]: {cmd} with args {args}")
    log.debug(f"[CMD]: {cmd} with stdin {stdin}")
    log.debug(f"[CMD]: {cmd} with timeout {timeout}, cwd: {cwd}")
    nm = nm or cmd
    stdout = stdout or ws / f'{nm}.stdout'
    stderr = stderr or ws / f'{nm}.stderr'

    full_env = {**os.environ, **(env or {})}

    with stdout.open('w') as fd_out, stderr.open('w') as fd_err:
        fd_in = Path(stdin).open('r') if stdin else None
        start_time = time.perf_counter_ns()
        exec_result = subprocess.run(
            [cmd, *args],
            stdout=fd_out,
            stderr=fd_err,
            stdin=fd_in,
            timeout=timeout,
            env=full_env,
            cwd=str(cwd) if cwd else None,
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


def print_suite_df(sdf: 'SuiteDef', colors: bool = True):
    tc = tcolors(colors)
    print(f"SUITE: [{tc.wrap(tc.GREEN, sdf.name)}] :: {sdf.desc}")
    for df in sdf.units:
        print(f"\tUNIT: [{tc.wrap(tc.GREEN, df.name)}]", f":: {df.desc}")
        for test in df.tests:
            print(
                f"\t- Test: [{tc.wrap(tc.CYAN, test.name)}] :: {test.desc} (Checks: {len(test.checks)})")
            for check in test.checks:
                print(
                    f"\t\t * Check: [{tc.wrap(tc.MAGENTA, check.name)}] :: {check.desc} "
                    f"[kind={check.kind}]"
                )
        print()


def print_suite_result(suite_res: 'SuiteRunResult', with_checks: bool = False,
                       colors: bool = True):
    tc = tcolors(colors)

    def _prk(r: 'GeneralResultType'):
        color = tc.RED if r.kind.is_fail() else tc.GREEN
        return tc.wrap(color, f"[{r.kind.value.upper()}]")

    def _p(r: 'GeneralResultType', t: str):
        return f"{_prk(r)} {t.capitalize()}: ({r.df.name}) :: {r.df.desc}"

    print(_p(suite_res, 'Suite'))
    for unit_res in suite_res.units:
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

    print(f"\nOVERALL RESULT: {_prk(suite_res)}\n")


def dump_junit_report(suite_res: 'SuiteRunResult', artifacts: Path) -> Optional[Path]:
    try:
        import junitparser
    except ImportError:
        LOG.warning("No JUNIT generated - junit parser is not installed")
        return None
    report_path = artifacts / 'junit_report.xml'
    LOG.info(f"[REPORT] Generating JUNIT report: {report_path}")
    junit_suites = junitparser.JUnitXml(suite_res.df.name)
    for unit_res in suite_res.units:
        unit_suite = junitparser.TestSuite(name=unit_res.df.name)
        for test_res in unit_res.tests:
            junit_case = junitparser.TestCase(
                name=test_res.df.desc,
                classname=test_res.df.unit.name + '/' + test_res.df.name,
                time=test_res.cmd_result.elapsed / 1000000.0 if test_res.cmd_result else 0
            )
            unit_suite.add_testcase(junit_case)
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
        junit_suites.add_testsuite(unit_suite)

    junit_suites.write(str(report_path))
    return report_path


def _resolve_def_file(name: str, root: Path, prefix=None) -> Optional['Path']:
    LOG.debug(f">>> Resolving: {name} in {root}")
    if '.' in name:
        pth = Path(name)
        return pth if pth.exists() else root / pth
    for ext in ('json', 'yml', 'yaml'):
        fname = f"{name}.{ext}"
        pth = root / fname
        if pth.exists():
            return pth
        if prefix:
            # Lets be compatible with kontr
            pth = root / f"{prefix}-{fname}"
            LOG.debug(f">>> Trying: {pth}")
            if pth.exists():
                return pth
    return None


## App params

class AppConfig(AsDict):
    def __init__(self, command: str, tests_dir: Path, data_dir: Path = None,
                 artifacts: Path = None):
        tests_dir = Path(tests_dir) if tests_dir else Path.cwd()
        self.command: str = command
        self.tests_dir: Path = Path(tests_dir).resolve()
        self.data_dir: Path = Path(data_dir) if data_dir else _resolve_data_dir(tests_dir)
        self.artifacts: Path = Path(artifacts) if artifacts else _make_artifacts_dir()

    def unit_workspace(self, suite: str, unit: str) -> Path:
        ws = self.suite_workspace(suite) / unit
        if not ws.exists():
            ws.mkdir(parents=True)
        return ws

    def suite_workspace(self, name: str) -> Path:
        return self.artifacts / name


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
        sub.add_argument('-S', '--suite', type=str,
                         help='Location or name of the suite/master definition file')
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
        return {'unit': _resolve_def_file(args.unit, cfg.tests_dir, prefix='unit')}
    if args.suite:
        return {'suite': _resolve_def_file(args.suite, cfg.tests_dir, 'master')}
    return {}


def cli_parse(args):
    cfg = _app_get_cfg(args)
    suite_df = _app_parse_suite(cfg, args)
    if args.output in ['json', 'j']:
        print(dump_json(suite_df))
    else:
        print_suite_df(suite_df)
    return True


def _app_parse_suite(cfg, args):
    defn = _resolve_definition(cfg, args)
    parser = DefinitionParser(cfg.tests_dir, data_dir=cfg.data_dir)
    return parser.parse(**defn)


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
    suite_df = _app_parse_suite(cfg, args)
    runner = DefinitionRunner(cfg)
    result = runner.run_suite(suite_df)
    print_suite_result(result)
    ws = cfg.suite_workspace(suite_df.name)
    print(f"SUITE WORKSPACE: {ws}")
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
