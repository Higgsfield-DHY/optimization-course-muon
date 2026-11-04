"""Persistent wall-clock accounting, shared by every GPU study operation."""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
import uuid

from filelock import FileLock


class BudgetExhausted(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path, seconds=28800., clock=None, wall_clock=None):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Budget must be finite and positive')
        self.path, self.limit = Path(path), float(seconds)
        self.clock, self.wall_clock = clock or time.perf_counter, wall_clock or time.time
        self.start = None
        self.shutdown_reserve=min(30.,self.limit*.1)
        self.operation_costs={}
        self.state = self._read()

    def _read(self):
        state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else dict(
            limit_seconds=self.limit, used_seconds=0., attempts=[], active=None)
        if state['limit_seconds'] != self.limit:
            raise ValueError('Existing budget limit cannot be changed')
        return state

    def _write(self):
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(self.path)

    @property
    def used_seconds(self):
        return self.state['used_seconds'] + (max(0., self.clock()-self.start) if self.start is not None else 0.)

    @property
    def remaining_seconds(self):
        return max(0., self.limit-self.used_seconds)

    def check(self,required_seconds=0.):
        if self.remaining_seconds <= self.shutdown_reserve+required_seconds:
            raise BudgetExhausted('GPU_BUDGET_EXHAUSTED')

    @contextmanager
    def operation(self,label):
        self.check(1.5*self.operation_costs.get(label,0.))
        started=self.clock()
        try:
            yield
        finally:
            self.operation_costs[label]=max(self.operation_costs.get(label,0.),self.clock()-started)

    def provenance(self):
        if self.start is None:
            raise ValueError('Accounting requires an active budget session')
        return dict(ledger_path=str(self.path.resolve()),session_id=self.state['active']['session_id'])

    def validate_provenance(self,provenance):
        if not provenance or provenance.get('ledger_path')!=str(self.path.resolve()):
            raise ValueError('Unaccounted evaluation cache')
        state=self._read()
        if not any(row.get('session_id')==provenance.get('session_id') and row['outcome']=='CLOSED'
                   for row in state['attempts']):
            raise ValueError('Evaluation cache lacks completed accounting provenance')

    @contextmanager
    def session(self, label):
        if self.start is not None:
            raise RuntimeError('Budget sessions cannot nest')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path)+'.lock', timeout=0):
            self.state = self._read()
            if self.state['active'] is not None:
                active = self.state['active']
                charged = max(0., self.wall_clock()-active['started_wall'])
                self.state['used_seconds'] += charged
                self.state['attempts'].append(dict(label=active['label'], seconds=charged,
                    outcome='RECOVERED_UNCLOSED_SESSION', reason='Conservative wall-clock charge'))
                self.state['active'] = None
                self._write()
            self.check()
            self.start = self.clock()
            session_id=uuid.uuid4().hex
            self.state['active'] = dict(label=label, started_wall=self.wall_clock(),session_id=session_id)
            self._write()
            outcome, reason = 'CLOSED', None
            try:
                yield self
            except BaseException as error:
                outcome, reason = 'EXCEPTION', f'{type(error).__name__}: {error}'
                raise
            finally:
                charged = max(0., self.clock()-self.start)
                self.state['used_seconds'] += charged
                self.state['attempts'].append(dict(label=label, seconds=charged, outcome=outcome, reason=reason,session_id=session_id))
                self.state['active'] = None
                self.start = None
                self._write()


def owning_study_ledger(config,run=None):
    settings=config['run']
    if settings.get('cohort')!='course_v1' or not settings['device'].startswith('cuda'):
        return None
    base=Path(run if run is not None else settings['output_root']).resolve()
    for parent in (base,*base.parents):
        if (parent/'study_state.json').is_file():
            return parent/'gpu_budget.json'
    raise ValueError('CUDA course run has no owning study manifest')


def require_study_accounting(config,budget,run=None):
    owner=owning_study_ledger(config,run)
    if owner is not None and (budget is None or budget.start is None or budget.path.resolve()!=owner.resolve() or budget.limit!=28800):
        raise ValueError('CUDA course work requires its owning active budget ledger')
    return owner
