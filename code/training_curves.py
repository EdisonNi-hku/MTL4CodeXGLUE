import json
from argparse import ArgumentParser
# import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default='training_state1.json')
    parser.add_argument("--bleu", type=int, default=0)
    args = parser.parse_args()

    with open(args.log_dir, 'r') as f:
        dic = json.load(f)

    bleu_em = dic['bleu_em']
    loss = dic['loss']
    print(dic['tr_nb'])
    print(dic['global_step'])

    if args.bleu == 0:
        print_dic = loss
    else:
        print_dic = bleu_em
    for task, results in print_dic.items():
        if 'identifier' in task or 'dataflow' in task:
            continue
        print(task)
        df = pd.DataFrame(results)
        print(df)
