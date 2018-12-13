import os
from gnas.common.folder_utils import make_dirs
from gnas.search_space.individual import Individual


def add_node(graph, node_id, label, shape='box', style='filled'):
    if label.startswith('Input'):
        color = 'skyblue'
    elif label.startswith('Output'):
        color = 'pink'
    elif 'Tanh' in label:
        color = 'yellow'
    elif 'ReLU' in label:
        color = 'orange'
    elif 'Sigmoid' in label:
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
    make_dirs(os.path.dirname(path))
    graph = pgv.AGraph(directed=True, strict=True,
                       fontname='Helvetica', arrowtype='open')  # not work?

    for i, oc in enumerate(ss.ocl):
        if i == 0:
            for j in range(oc.get_n_inputs()):
                add_node(graph, j, 'Input' + str(j))
            add_node(graph, j + 1, type(oc))
        else:
            print("a")

    # for i in range(ss.n_inputs):
    #     add_node(graph, i, 'Input')
    #
    # for i, nc in enumerate(individual.generate_node_config()):
    #     if i >= ss.n_nodes:
    #         add_node(graph, i + ss.n_inputs, 'Output-' + str(nc))
    #     else:
    #         add_node(graph, i + ss.n_inputs, 'Node-' + str(nc))
    #     # for j, v in enumerate(nc.connection_index):
    #     #     if v > 0:
    #     graph.add_edge(nc.connection_index, i + ss.n_inputs)
    graph.layout(prog='dot')
    graph.draw(path)
