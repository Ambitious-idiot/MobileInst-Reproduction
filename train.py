import torch
from torch.optim import AdamW
from detectron2.solver import WarmupMultiStepLR
import os
import time

import mobileinst
import utils


def train(args):
    utils.set_seed(3407)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader = mobileinst.build_train_dataloader(args.root, args.n_cls, args.batch_size)

    backbone = mobileinst.backbones.get_topformer(device)
    model = mobileinst.MobileInst(backbone, args.channels, args.dim, args.key_dim, args.num_heads,
                                  args.n_cls, args.mlp_ratios, args.attn_ratios).to(device)
    matcher = mobileinst.MobileInstMatcher(args.alpha, args.beta, args.dim, device)
    criterion = mobileinst.MobileInstCriterion(
        (args.ce_weight, args.mask_weight, args.dice_weight, args.obj_weight),
        args.n_cls, matcher, device)
    optimizer = AdamW(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = WarmupMultiStepLR(optimizer, args.steps)
    start_epoch = -1

    if args.resume:
        print('Load parameters')
        checkpoint = os.path.join(args.checkpoint_root, args.checkpoint)
        assert os.path.exists(checkpoint)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    print('Training starts')
    for epoch in range(start_epoch+1, args.max_iter):
        loss_stats = {"loss_ce": 0., "loss_mask": 0.,
                      "loss_dice": 0., "loss_objectness": 0.}
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            image = batch['image'].to(device)
            outputs = model(image)
            losses = criterion(outputs, batch)
            total_loss = 0
            for name, loss in losses.items():
                total_loss += loss
                loss_stats[name] += loss.item()
            total_loss.backward()
            if i % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        log = 'Epoch%04d[%s]' % (epoch+1, time.strftime('%m-%d %H:%M:%S'))
        for k, v in loss_stats.items():
            log += '%s=%.4f ' % (k, v)
        print(log)
        if epoch % 50 == 49:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_root, 'mobileinst%04d.pth' % (epoch+1)))


if __name__ == '__main__':
    args = utils.parse_args()
    train(args)
