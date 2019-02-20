import os
import numpy as np

from gnas.search_space.individual import Individual, MultipleBlockIndividual
from gnas.search_space.operation_space import RnnNodeConfig, RnnInputNodeConfig, CnnNodeConfig


def add_node(graph, node_id, label, shape='box', style='filled'):
    if label.startswith('Input') or label.startswith('x'):
        color = 'skyblue'
    elif label.startswith('Output'):
        color = 'pink'
    elif 'Tanh' in label or 'Add' in label or 'Concat' in label:
        color = 'yellow'
    elif 'ReLU' in label or 'Dw' in label or 'Conv' in label:
        color = 'orange'
    elif 'Sigmoid' in label or 'Identity' in label:
        color = 'greenyellow'
    elif label == 'avg':
        color = 'seagreen3'
    else:
        color = 'white'

    if not any(label.startswith(word) for word in ['x', 'avg', 'h']):
        label = f"{label}\n({node_id})"

    graph.add_node(
        node_id, label=label, color='black', fillcolor=color,
        shape=shape, style=style,
    )


def _draw_individual(ocl, individual, path=None):
    import pygraphviz as pgv
    graph = pgv.AGraph(directed=True, layout='dot')  # not work?

    ofset = len(ocl[0].inputs)
    for i in range(len(ocl[0].inputs)):
        add_node(graph, i, 'x[' + str(i) + ']')

    input_list = []

    for i, (oc, op) in enumerate(zip(individual.generate_node_config(), ocl)):
        if isinstance(op, CnnNodeConfig):
            input_a = oc[0]
            input_b = oc[1]
            input_list.append(input_a)
            input_list.append(input_b)
            op_a = oc[4]
            op_b = oc[5]
            add_node(graph, (i + ofset) * 10, ocl[i].op_list[op_a])
            add_node(graph, (i + ofset) * 10 + 1, ocl[i].op_list[op_b])
            graph.add_edge(input_a, (i + ofset) * 10)
            graph.add_edge(input_b, (i + ofset) * 10 + 1)
            add_node(graph, (i + ofset), 'Add')
            graph.add_edge((i + ofset) * 10, (i + ofset))
            graph.add_edge((i + ofset) * 10 + 1, (i + ofset))
            c = graph.add_subgraph([(i + ofset) * 10, (i + ofset) * 10 + 1, (i + ofset)],
                                   name='cluster_block:' + str(i), label='Block ' + str(i))
            # c.attr(label='block:'+str(i))

        elif isinstance(op, RnnInputNodeConfig):
            op_type = op.non_linear_list[oc]
            add_node(graph, (i + ofset), op_type)
            graph.add_edge(0, (i + ofset))
            graph.add_edge(1, (i + ofset))
            input_list.append(0)
            input_list.append(1)

        elif isinstance(op, RnnNodeConfig):
            op_type = op.non_linear_list[oc[-1]]
            add_node(graph, (i + ofset), op_type)
            graph.add_edge(oc[0], (i + ofset))
            input_list.append(oc[0])
        else:
            raise Exception('unkown node type')
    input_list = np.unique(input_list)
    op_inputs = [int(i) for i in np.linspace(ofset, ofset + individual.get_n_op() - 1, individual.get_n_op()) if
                 i not in input_list]
    concat_node = i + 1 + ofset
    add_node(graph, concat_node, 'Concat')
    for i in op_inputs:
        graph.add_edge(i, concat_node)
    graph.layout(prog='dot')
    if path is not None:
        graph.draw(path + '.png')



def draw_cell(ocl, individual):
    _draw_individual(ocl, individual, path=None)


def draw_network(ss, individual, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(individual, Individual):
        _draw_individual(ss.ocl, individual, path)
    elif isinstance(individual, MultipleBlockIndividual):
        [_draw_individual(ocl, inv, path + str(i)) for i, (inv, ocl) in
         enumerate(zip(individual.individual_list, ss.ocl))]
