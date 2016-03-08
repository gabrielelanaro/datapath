from .core import Call

from graphviz import Digraph
from itertools import count
import matplotlib.pyplot as plt


def plot_trail(trail):

    graph = Digraph()
    graph.attr('node', shape='box', fontsize='12', penwidth='1.0', color='#888888', style='rounded')
    
    c = count()

    def _make_graph(trail):
        if isinstance(trail, Call):
            # We need to make a node
            node = str(c.next())
            graph.node(node, format_args(trail))

            for a in trail.args:
                if isinstance(a, Call):
                    parent = _make_graph(a)
                    graph.edge(parent, node)

            for v in trail.kwargs.values():
                if isinstance(v, Call):
                    parent = _make_graph(v)
                    graph.edge(parent, node)

            return node

    _make_graph(trail)
    return graph


def format_args(call):
    arg_style = r'{}'
    args = ', '.join(' _ ' if isinstance(a, Call) else arg_style.format(repr(a))
                     for a in call.args)
    kwargs = ','.join(k + '=' + str(v)
                      for k, v in call.kwargs.items())

    if args == '' and kwargs == '':
        tpl = '{}{}{}'

    elif args == '' and kwargs != '' or kwargs == '' and args != '':
        tpl = "{}({}{})"

    else:
        tpl = "{}({}, {})"

    return tpl.format(call.func, args, kwargs)
