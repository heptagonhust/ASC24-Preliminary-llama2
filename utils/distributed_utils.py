from typing import List, Dict

import ray
from ray.util.placement_group import (
    PlacementGroup
)
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy
)

def get_node_id():
    node_id = ray.get_runtime_context().get_node_id()
    return node_id

def get_node_id_of_pgs(pg_list: List[PlacementGroup]):
    node_id_to_pg: Dict[int: int] = {}
    for pg_index, pg in enumerate(pg_list):
        node_id = ray.remote(num_cpus=1,
                             scheduling_strategy=PlacementGroupSchedulingStrategy(
                                 placement_group=pg,
                                 placement_group_bundle_index=0,
                             ))(get_node_id).remote()
        node_id_to_pg[ray.get(node_id)] = pg_index
    return node_id_to_pg
