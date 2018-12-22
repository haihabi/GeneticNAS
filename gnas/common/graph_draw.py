import os
import numpy as np

from gnas.search_space.individual import Individual


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


def draw_network(ss, individual: Individual, path):
    import pygraphviz as pgv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    graph = pgv.AGraph(directed=True, strict=True,
                       fontname='Helvetica', arrowtype='open')  # not work?

    for i in range(2):
        add_node(graph, i, 'x[' + str(i) + ']')

    input_list = []
    for i, oc in enumerate(individual.generate_node_config()):
        input_a = oc[0]
        input_b = oc[1]
        input_list.append(input_a)
        input_list.append(input_b)
        op_a = oc[4]
        op_b = oc[5]
        add_node(graph, (i + 2) * 10, ss.ocl[i].op_list[op_a])
        add_node(graph, (i + 2) * 10 + 1, ss.ocl[i].op_list[op_b])
        graph.add_edge(input_a, (i + 2) * 10)
        graph.add_edge(input_b, (i + 2) * 10 + 1)
        add_node(graph, (i + 2), 'Add')
        graph.add_edge((i + 2) * 10, (i + 2))
        graph.add_edge((i + 2) * 10 + 1, (i + 2))
    input_list = np.unique(input_list)
    op_inputs = [int(i) for i in np.linspace(2, 2 + individual.get_n_op() - 1, individual.get_n_op()) if
                 i not in input_list]
    concat_node = i + 3
    add_node(graph, concat_node, 'Concat')
    for i in op_inputs:
        graph.add_edge(i, concat_node)
    graph.layout(prog='dot')
    graph.draw(path)
