import os
configs = {}
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    configs[module[:-3]] = __import__(module[:-3], locals(), globals()).training_params

