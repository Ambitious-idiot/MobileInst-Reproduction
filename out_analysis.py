import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='nohup.out', help='Path to analyze')
parser.add_argument('--db', type=bool, default=False, help='Convert dB')
parser.add_argument('--start', type=int, default=0, help='Start epoch')
args = parser.parse_args()

epoch = 1
losses = {
    'loss_ce': [],
    'loss_dice': [],
    'loss_mask': [],
    'loss_objectness': []
}
with open(args.path) as f:
    for line in f.readlines():
        if not line.startswith('Epoch'):
            continue
        line = line[5:].strip()
        e, line = line.split('[')
        e = int(e)
        if e < epoch:
            for key in losses.keys():
                losses[key] = losses[key][:e]
        epoch = e
        line = line.split(']')[-1]
        results = line.split(' ')
        for result in results:
            k, v = result.split('=')
            losses[k].append(float(v))


root = os.path.dirname(__file__)
for k, v in losses.items():
    plt.figure()
    plt.grid()
    v = v[args.start:]
    v = np.array(10 * np.log10(v)) if args.db else v
    plt.plot(v)
    if args.db:
        plt.ylabel('dB')
    else:
        plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(k)
    plt.savefig(os.path.join(root, 'out', 'losses', '%s.jpg'%k))
plt.figure()
tot_loss = sum([np.array(v) for v in losses.values()])[args.start:]
tot_loss = np.array(10 * np.log10(tot_loss)) if args.db else tot_loss
plt.plot(tot_loss)
if args.db:
    plt.ylabel('dB')
else:
    plt.ylabel('loss')
plt.title('loss_total')
plt.savefig(os.path.join(root, 'out', 'losses', 'loss_total.jpg'))
