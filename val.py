import torch
import os
import mobileinst
import utils


device = torch.device('cpu')
args = utils.parse_args()
backbone = mobileinst.backbones.get_topformer(device)
model = mobileinst.MobileInst(backbone, args.channels, args.dim, args.num_kernels, args.key_dim, args.num_heads,
                              args.n_cls, args.mlp_ratios, args.attn_ratios).to(device)

checkpoint = os.path.join(args.checkpoint_root, args.checkpoint)
assert os.path.exists(checkpoint)
checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint['model'])

evaluator = mobileinst.Evaluator(args, model, device)

# AP
res_file = evaluator.gen_result_json()
print(evaluator.evaluate(res_file))

# for i in range(len(evaluator)):
#     result = evaluator.inference(i)
#     if 'pred_masks' in result.keys():
#         mask = utils.fuse_masks(result['pred_masks'], True)
#         mask.save(os.path.join(args.out_root, 'pred_masks', 'mask%r.jpg' % i))
