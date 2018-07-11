import tensorflow as tf
from tensorflow.python.framework import ops
from collections import defaultdict
import pandas as pd
import os
inference_ops = "Relu,Relu6,Prelu,Elu,Add,Conv2D, MatMul, VariableV2, MaxPool2D,AvgPool2D"
exclude_in_name = ["gradients", "Initializer", "Regularizer", "AssignMovingAvg", "Momentum", "BatchNorm"]


class _ModelStats(tf.train.SessionRunHook):
    """Logs model stats to a csv."""

    def __init__(self, scope_name, path,batch_size):
        self.scope_name = scope_name
        self.batch_size = batch_size
        self.path = path

    def after_create_session(self, session, coord):
        graph = tf.get_default_graph()
        operations = graph.get_operations()
        biases = defaultdict(lambda: None)
        stat_dict = defaultdict(lambda: {"params":0,"maccs":0})
        for tensor in operations:
            name = tensor.name
            if not self.scope_name in tensor.name or any(exclude_name in name for exclude_name in exclude_in_name):
                continue
            if not tensor.type in inference_ops:
                continue
            base_name = "/".join(name.split("/")[:-1])
            if name.endswith("weights"):
                if any(base_name + "BatchNorm" in operation.name for operation in operations) or any(
                        base_name + "biases" in operation.name for operation in operations):
                    biases[base_name] = int(sizes[-1])
                shape = tensor.node_def.attr["shape"].shape.dim
                sizes = [int(size.size) for size in shape]
                params = 1
                for dim in sizes:
                    params = params * dim
                if biases[base_name]:
                    params = params + biases[base_name]
                stat_dict[base_name]["params"] = params
            elif tensor.type == "Add":
                flops = ops.get_stats_for_node_def(graph, tensor.node_def, 'flops').value
                if flops is not None:
                    stat_dict[name]["maccs"] = flops / 2 / self.batch_size
            else:
                flops = ops.get_stats_for_node_def(graph, tensor.node_def, 'flops').value
                if flops is not None:
                    if biases[base_name]:
                        stat_dict[base_name]["maccs"] = int(flops / 2 / self.batch_size) + biases[base_name]
                    else:
                        stat_dict[base_name]["maccs"] = int(flops/2/self.batch_size)
        total_params = 0
        total_maccs = 0
        for key,stat in stat_dict.iteritems():
            total_maccs += stat["maccs"]
            total_params += stat["params"]
        stat_dict["total"] = {"maccs":total_maccs,"params":total_params}
        df = pd.DataFrame.from_dict(stat_dict, orient='index')
        df.to_csv(os.path.join(self.path,'model_stats.csv'))
