import os
import dask.threaded
import dis
import dill
import joblib
import marshal

from dask.compatibility import apply


class DataCache:

    def __init__(self, directory='./dc'):
        if not os.path.exists(directory):
            os.mkdir(directory)

        self.directory = directory
        self.graph = {}
        self.step_graph = {}
        self.observed = []

    def step(self, target, *args, **kwargs):
        name = kwargs.get('name')

        if name is None:
            name = target.__name__

        return Step(self, name, target, args, kwargs)

    def record(self, target, *args, **kwargs):
        processed_args = []
        for arg in args:
            if isinstance(arg, Step):
                arg = arg.get()

            processed_args.append(arg)
        result = target(*processed_args, **kwargs)

        self.observed.append((target.__name__, args, kwargs, result))
        return result

    def summary(self):
        # We need to infer the pipeline from our graph.
        lines = []
        for name, args, kwargs, result in self.observed:
            args = tuple(a if not isinstance(a, Step)
                         else a.name for a in args)

            lines.append(name + str(args) + ' = ' + str(result))

        return '\n'.join(lines)

    def store(self, name, value):
        return joblib.dump(value, os.path.join(self.directory, name))

    def load(self, name):
        return joblib.load(os.path.join(self.directory, name))

    def load_hash(self, name):
        hash_file = os.path.join(self.directory, name + '.hash')
        if not os.path.exists(hash_file):
            return None
        else:
            return open(hash_file).read()

    def store_hash(self, name, hash):
        hash_file = os.path.join(self.directory, name + '.hash')
        with open(hash_file, 'w') as fd:
            fd.write(hash)


class Step:

    def __init__(self, dc, name, target, args, kwargs):
        self.dc = dc
        self.target = target
        self.name = name
        self.args = args
        self.kwargs = kwargs

        # Create a new entry in the graph
        self.recompute()
        self.dc.step_graph[self.name] = self

    def recompute(self):
        self.dc.graph[self.name] = (apply_with_kwargs, self.target,
                                    list(self.args), list(self.kwargs))

    def step(self, target, *args, **kwargs):

        name = kwargs.get('name')
        if name is None:
            name = target.__name__

        # We prepend our result as our first argument to allow easy chaining
        args = (self.name,) + args

        return Step(self.dc, self.name + '->' + name, target, args, kwargs)

    def get(self):
        result = dask.threaded.get(self.dc.graph, self.name)
        return result

    def checkpoint(self):
        hash_ = self.hash()
        if hash_ != self.dc.load_hash(self.name):
            # If hash has changed, we write stuff to disk
            self.recompute()
            result = self.get()
            self.dc.store(self.name + '.store', result)
            self.dc.store_hash(self.name, hash_)

        # We replace the calculation with the cached value
        self.dc.graph[self.name] = (self.dc.load, self.name + '.store')
        return self

    def previous(self):
        return self.dc.step_graph['->'.join(self.name.split('->')[:-1])]

    def hash(self):
        uniquity = (self.name, self.args, self.kwargs,
                    marshal.dumps(self.target.__code__))

        if '->' in self.name:
            previous_hash = self.previous().hash()
        else:
            previous_hash = ''

        return previous_hash + joblib.hash(uniquity)


def apply_with_kwargs(function, args, kwargs):
    return function(*args, **dict(kwargs))
