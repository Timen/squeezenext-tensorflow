from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.framework import ops
from collections import defaultdict
import pandas as pd
import os
supported_stat_ops = "Conv2D, MatMul, VariableV2, MaxPool,AvgPool,Add"
exclude_in_name = ["gradients", "Initializer", "Regularizer", "AssignMovingAvg", "Momentum", "BatchNorm"]


class ModelStats(tf.train.SessionRunHook):
    """Logs model stats to a csv."""

    def __init__(self, scope_name, path,batch_size):
        """
        Set class variables
        :param scope_name:
            Used to filter for tensors which name contain that specific variable scope
        :param path:
            path to model dir
        :param batch_size:
            batch size during training
        """
        self.scope_name = scope_name
        self.batch_size = batch_size
        self.path = path
        self.inc_bef =0
        self.inc_after = 0

    def begin(self):
        """
            Method to output statistics of the model to an easy to read csv, listing the multiply accumulates(maccs) and
            number of parameters, in the model dir.
        :param session:
            Tensorflow session
        :param coord:
            unused
        """
        # get graph and operations
        graph = tf.get_default_graph()
        operations = graph.get_operations()
        # setup dictionaries
        biases = defaultdict(lambda: None)
        stat_dict = defaultdict(lambda: {"params":0,"maccs":0,"adds":0, "comps":0})

        # iterate over tensors
        for tensor in operations:
            name = tensor.name
            # check is scope_name is in name, or any of the excluded strings
            if not self.scope_name in name or any(exclude_name in name for exclude_name in exclude_in_name):
                continue
            # Check if type is considered for the param and macc calcualtion
            if not tensor.type in supported_stat_ops:
                continue

            base_name = "/".join(name.split("/")[:-1])

            if name.endswith("weights"):
                shape = tensor.node_def.attr["shape"].shape.dim
                sizes = [int(size.size) for size in shape]
                if any(base_name + "/BatchNorm" in operation.name for operation in operations) or any(
                        base_name + "/biases" in operation.name for operation in operations):
                    biases[base_name] = int(sizes[-1])
                params = 1
                for dim in sizes:
                    params = params * dim

                if biases[base_name] is not None:
                    params = params + biases[base_name]
                stat_dict[base_name]["params"] = params
            elif tensor.type == "Add":
                flops = ops.get_stats_for_node_def(graph, tensor.node_def, 'flops').value
                if flops is not None:
                    stat_dict[name]["adds"] = flops  / self.batch_size
            elif tensor.type == "MaxPool":
                flops = ops.get_stats_for_node_def(graph, tensor.node_def, 'comps').value
                if flops is not None:
                    stat_dict[name]["comps"] = flops / self.batch_size
            elif tensor.type == "AvgPool":
                flops = ops.get_stats_for_node_def(graph, tensor.node_def, 'flops').value
                if flops is not None:
                    stat_dict[name]["adds"] = flops / self.batch_size
            elif  tensor.type == "MatMul" or tensor.type == "Conv2D":
                flops = ops.get_stats_for_node_def(graph, tensor.node_def, 'flops').value
                if flops is not None:
                    stat_dict[base_name]["maccs"] += int(flops / 2 / self.batch_size)
            elif name.endswith("biases"):
                pass
            else:
                print(name,tensor.type)
                exit()
        total_params = 0
        total_maccs = 0
        total_comps = 0
        total_adds = 0
        for key,stat in stat_dict.iteritems():
            total_maccs += stat["maccs"]
            total_params += stat["params"]
            total_adds += stat["adds"]
            total_comps += stat["comps"]
        stat_dict["total"] = {"maccs":total_maccs,"params":total_params, "adds":total_adds, "comps":total_comps}
        df = pd.DataFrame.from_dict(stat_dict, orient='index')
        df.to_csv(os.path.join(self.path,'model_stats.csv'))
