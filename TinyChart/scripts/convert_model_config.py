import argparse
import json
import os

def convert_config(path):
    config_path = path+'/config.json'
    config = json.load(open(config_path, 'r'))
    assert os.path.isdir(path+'/vision_tower')
    try:
        os.symlink(path+'/vision_tower', path+'/siglip')
    except:
        pass
    config['mm_vision_tower'] = path+'/siglip'
    json.dump(config, open(config_path, 'w'), indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)

    args = parser.parse_args()

    if args.input[0] != '/':
        args.input = os.getcwd() + '/' + args.input

    if os.path.isdir(args.input+'/vision_tower'):
        convert_config(args.input)