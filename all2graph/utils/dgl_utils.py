import platform
system = platform.system()
print('system: {}'.format(system))
if 'windows' not in platform.system().lower():
    import os
    import json
    print(os.environ)
    old_home = os.environ['HOME']
    new_home = os.getcwd()
    os.environ['HOME'] = new_home
    print(os.environ['HOME'])
    import dgl
    config_path = os.path.join(new_home, '.dgl', 'config.json')
    print('all2graph has moved dgl config.json to {}'.format(config_path))
    with open(config_path, 'r') as file:
        print(json.load(file))
    os.environ['HOME'] = old_home
else:
    import dgl
