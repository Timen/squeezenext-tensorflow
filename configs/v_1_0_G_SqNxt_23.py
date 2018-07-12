from __future__ import absolute_import

training_params = {
    "base_lr": 0.4,
    "image_size":227,
    "block_defs":[(32,6,1),(64,6,2),(128,8,2),(256,1,2)],
    "input_def":(64,(7,7),2),
    "num_classes":1000,
    "group_size":2
}