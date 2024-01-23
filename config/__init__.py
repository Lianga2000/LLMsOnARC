# import sys
#
# from omegaconf import OmegaConf
# CONFIG_LOCATION = r'config/config.yaml'
# conf = OmegaConf.load(CONFIG_LOCATION)
# second_conf = OmegaConf.load(r'config/code_llama_cot_config.yaml')
# cli_conf = OmegaConf.from_cli(sys.argv[1:])
#
# config = OmegaConf.merge(conf,second_conf, cli_conf)

import os
from omegaconf import OmegaConf

# The default
config_names = os.getenv('CONFIG_NAMES', None)
print(f'CONFIG_NAMES: {config_names}')
base_path = r'config'
configs = [OmegaConf.load(fr'{base_path}/config.yaml')]
if config_names is not None:
    for config_name in config_names.split(','):
        configs.append(OmegaConf.load(f'{base_path}/{config_name.strip()}.yaml'))
command_line_conf = OmegaConf.from_cli()
configs.append(command_line_conf)
# unsafe_merge makes the individual configs unusable, but it is faster
config = OmegaConf.unsafe_merge(*configs)

if __name__ == '__main__':
    print(config)
