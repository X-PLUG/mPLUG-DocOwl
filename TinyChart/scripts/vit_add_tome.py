import json
import argparse

def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data, data_path):
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--image_size', type=int, default=768)
    parser.add_argument('--tome_r', type=int, default=84)

    args = parser.parse_args()

    config = read_json(args.path+'/config.json')
    config['use_tome'] = True
    config['image_size'] = args.image_size
    config['tome_r'] = args.tome_r
    write_json(config, args.path+'/config.json')

    