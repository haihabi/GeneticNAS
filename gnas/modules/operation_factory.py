from gnas.search_space.operation_space import RnnNodeConfig, RnnInputNodeConfig
from gnas.modules.node_module import RnnNodeModule, RnnInputNodeModule

__module_dict__ = {RnnNodeConfig: RnnNodeModule,
                   RnnInputNodeConfig: RnnInputNodeModule}


def get_module(node_config):
    m = __module_dict__.get(type(node_config))
    if m is None:
        raise Exception('Can\'t find module named:' + node_config)
    return m(node_config)
