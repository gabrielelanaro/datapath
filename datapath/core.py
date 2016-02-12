import os
import dask.threaded
from dask.compatibility import apply
class DataCache:
    
    def __init__(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        
        self.graph = {}
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
            args = tuple(a if not isinstance(a, Step) else a.name for a in args)
            
            lines.append(name + str(args) + ' = ' + str(result))
        
        return '\n'.join(lines)

class Step:
    
    def __init__(self, dc, name, target, args, kwargs):
        self.dc = dc
        self.target = target
        self.name = name
        self.args = args
        self.kwargs = kwargs
        
        # Create a new entry in the graph
        self.dc.graph[self.name] = (apply_with_kwargs, self.target, list(args), list(kwargs))

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


def apply_with_kwargs(function, args, kwargs):
    return function(*args, **dict(kwargs))
    
