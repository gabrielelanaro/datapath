import os
import dask.threaded
import dis
import joblib
import types
import shelve

from collections import namedtuple
from operator import getitem
from dask.compatibility import apply
from .utils import hashabledict


Call = namedtuple('Call', ['func', 'args', 'kwargs'])


def resolve_args(args):
    return tuple(a if not isinstance(a, Step) else a.trail for a in args)


def resolve_kwargs(kwargs):
    return {k: v if not isinstance(v, Step) else v.trail for k, v in kwargs.items()}

def fmt_call(call):
    
class DataCache:

    def __init__(self, directory='./dc'):
        if not os.path.exists(directory):
            os.mkdir(directory)

        self.directory = directory
        self.graph = {}
        self.step_graph = {}

    def step(self, target, *args, **kwargs):
        name = target.__name__
        args = resolve_args(args)
        kwargs = resolve_kwargs(kwargs)

        trail = Call(name, args, hashabledict(kwargs))
        return Step(self, trail, target, args, kwargs)

    def record(self, trail, value):
        recorded = shelve.open(os.path.join(self.directory, 'recorded.shelve'))
        recorded[make_path(trail)] = {'trail': trail,
                                           'value': value}
        return value

    def _make_summary(self):
        recorded = shelve.open(os.path.join(self.directory, 'recorded.shelve'))
        r = []
        for v in recorded.values():
            r.append(repr(v['trail']))
            r.append(repr(v['value']))
            r.append('--\n')
            
        return '\n'.join(r)
                
    def summary(self):
        return self._make_summary()

    def store(self, trail, value):
        return joblib.dump(value, os.path.join(self.directory, make_path(trail)))

    def load(self, trail):
        return joblib.load(os.path.join(self.directory, make_path(trail)))

    def load_hash(self, trail):
        hash_file = os.path.join(self.directory, make_path(trail) + '.hash')
        if not os.path.exists(hash_file):
            return None
        else:
            return open(hash_file).read()

    def store_hash(self, trail, hash):
        hash_file = os.path.join(self.directory, make_path(trail) + '.hash')
        with open(hash_file, 'w') as fd:
            fd.write(hash)

    def load_meta(self, step, meta):
        '''Load some metadata for a certain step'''
        meta_file = os.path.join(
            self.directory, make_path(step.trail) + '.shelve')

        shv = shelve.open(meta_file)
        if meta not in shv:
            return None
        else:
            return shv[meta]

    def store_meta(self, step, meta, value):
        '''Store metadata for a certain step'''
        meta_file = os.path.join(
            self.directory, make_path(step.trail) + '.shelve')
        shv = shelve.open(meta_file)
        shv[meta] = value


def make_path(name_tuple):
    return joblib.hash(name_tuple)


class Step:

    def __init__(self, dc, trail, target, args, kwargs):
        self.dc = dc
        self.target = target
        self.trail = trail
        self.args = args
        self.kwargs = kwargs

        self.dc.step_graph[self.trail] = self
        self.recompute()

        self._counter = 0

    def recompute(self):
        self.dc.graph[self.trail] = (apply_with_kwargs, self.target,
                                     list(self.args), list(self.kwargs))

    def step(self, target, *args, **kwargs):
        name = target.__name__

        # We prepend our result as our first argument to allow easy chaining
        args = (self.trail,) + args

        # Our trail gets expanded
        trail = Call(name, args, hashabledict(kwargs))

        return Step(self.dc, trail, target, args, kwargs)

    def get(self):
        result = dask.threaded.get(self.dc.graph, self.trail)
        return result

    def changed(self):
        return self.hash() != self.dc.load_hash(self.trail)

    def checkpoint(self, recompute=False):
        hash_ = self.hash()

        if hash_ != self.dc.load_hash(self.trail) or recompute:
            # If hash has changed, we write stuff to disk
            self.recompute()
            result = self.get()
            self.dc.store(self.trail + ('.store',), result)
            self.dc.store_hash(self.trail, hash_)

        # We replace the calculation with the cached value
        self.dc.graph[self.trail] = (self.dc.load, self.trail + ('.store',))
        return self

    def previous(self):
        for a in self.trail.args:
            if isinstance(a, Call):
                yield self.dc.step_graph[a]

        for k, v in self.trail.kwargs:
            if isinstance(v, Call):
                yield self.dc.step_graph[v]

    def has_deps(self):
        return (any(isinstance(a, Call) for a in self.trail.args)
                or any(isinstance(v, Call) for v in self.trail.kwargs.values()))

    def hash(self):

        if isinstance(self.target, types.BuiltinFunctionType):
            bytecode = None
            consts = None
        else:
            bytecode = self.target.__code__.co_code
            consts = self.target.__code__.co_consts

        uniquity = (self.trail, self.args, self.kwargs,
                    bytecode,
                    consts)

        if self.has_deps():
            previous_hash = ''.join(p.hash() for p in self.previous())
        else:
            previous_hash = ''

        return previous_hash + joblib.hash(uniquity)

    def store_meta(self, meta, value):
        self.dc.store_meta(self, meta, value)

    def load_meta(self, meta):
        return self.dc.load_meta(self, meta)

    def record(self, target, *args, **kwargs):
        final = self.step(target, *args, **kwargs)
        return self.dc.record(final.trail, final.get())

    def prepr(self):
        '''Pretty representation'''
        path = []

        func_name = self.trail.func

        args = ', '.join('*' if isinstance(a, Call) else str(a)
                         for a in self.trail.args)
        kwargs = ','.join(k + '=' + str(v)
                          for k, v in self.trail.kwargs.items())

        if args == '' and kwargs == '':
            tpl = "{}{}{}"

        elif args == '' and kwargs != '' or kwargs == '' and args != '':
            tpl = "{}({}{})"

        else:
            tpl = "{}({}, {})"

        path.append(tpl.format(self.trail.func, args, kwargs))
        for p in self.previous():
            path.append(p.prepr())

        return '<-'.join(path)

    def length(self):

        if self.load_meta('length') is None or self.changed():
            length = len(self.get())
            self.store_meta('length', length)
        else:
            length = self.load_meta('length')

        return length

    def __getitem__(self, item):
        if item >= self.length():
            raise IndexError()
        else:
            return self.step(getitem, item)


def apply_with_kwargs(function, args, kwargs):
    return function(*args, **dict(kwargs))
    
    
