from __future__ import absolute_import

# Some parameters will be divided or multiplied by 4 to compensate for the reduced batch size
# as the original used batch size 1024 and the gtx1080ti can only fit a batch size 256.

training_params = {
    "base_lr":0.4/4,
    "warmup_iter":780*4,
    "warmup_start_lr":0.1/4,
    "image_size":227,
    "block_defs":[(32,6,1),(64,6,2),(128,8,2),(256,1,2)],
    "input_def":(64,(7,7),2),
    "num_classes":1000,
    "groups": 1
}