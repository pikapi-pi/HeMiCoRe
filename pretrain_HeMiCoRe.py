import os
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json
import dhg
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import WSIWithCluster, mixup, get_feats
from utils.general import AverageMeter, CSVWriter, EarlyStop, increment_path, BestVariable, init_seeds, save_checkpoint, load_json
from models import rlmil, cl
from utils.losses import NT_Xent
from HGNN import H_GNN


def create_save_dir(args):
    """
    Create directory to save experiment results by global arguments.
    :param args: the global arguments
    """
    dir_data_split = f'{Path(args.data_split_json).stem}'
    dir1 = f'{args.dataset}_np_{args.feat_size}'
    dir2 = f'pretrain'
    murcl_setting = [
        f'T{args.T}',
        f'pd{args.projection_dim}',
        f'as{args.action_std}',
        f'pg{args.ppo_gamma}',
        f'tau{args.temperature}',
        f'alpha{args.alpha}',
    ]
    dir3 = '_'.join(murcl_setting)
    dir4 = args.arch
    if args.arch in ['HGNN']:
        arch_setting = [
            f'depth_{args.GCN_depth}',
            f'cluster_{args.Cluster}'
        ]
    else:
        raise ValueError()
    dir5 = '_'.join(arch_setting)
    dir6 = f'exp'
    if args.save_dir_flag is not None:
        dir6 = f'{dir6}_{args.save_dir_flag}'
    dir7 = f'seed{args.seed}'
    dir8 = f'stage_{args.train_stage}'
    args.save_dir = str(Path(args.base_save_dir) / dir_data_split / dir1 / dir2 / dir3 / dir4 / dir5 / dir6 / dir7 / dir8)
    print(f"save_dir: {args.save_dir}")


def get_datasets(args: object) -> object:
    indices = load_json(args.data_split_json)
    train_set = WSIWithCluster(
        data_csv=args.data_csv,
        indices=indices['train'] + indices['valid'],
        shuffle=True,
        preload=args.preload,
    )
    args.num_clusters = train_set.num_clusters
    return train_set, train_set.patch_dim, len(train_set)


def create_model(args, dim_patch):
    print(f"Creating model {args.arch}...")
    # Create raw model/ MIL aggregator
    if args.arch == 'HGNN':
        args.feature_num = dim_patch
        print(args.GCN_hidden_dim, dim_patch)
        model = H_GNN.HGNN(
            in_dim=dim_patch,
            n_hid=args.GCN_hidden_dim,
            out_dim=dim_patch,
            drop_rate=args.GCN_dropout
        )
    else:
        raise NotImplementedError(f'args.arch error, {args.arch}. ')
    # Wrapping with CL structure
    if args.arch == 'HGNN':
        model = cl.CL(model, projection_dim=args.projection_dim, n_features=model.in_features)
    else:
        raise NotImplementedError
    fc = rlmil.Full_layer(args.feature_num, args.fc_hidden_dim, args.fc_rnn, args.projection_dim)#the project head, for changing v to p
    ppo = None

    # Load checkpoint
    if args.train_stage == 1:
        pass
    elif args.train_stage == 2:
        # if not specify the checkpoint path, use the default path produced from previous stage path.
        if args.checkpoint is None:
            args.checkpoint = str(Path(args.save_dir).parent / 'stage_1' / 'model_best.pth.tar')
        assert Path(args.checkpoint).exists(), f"{args.checkpoint} is not exist!"

        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        fc.load_state_dict(checkpoint['fc'])

        state_dim = args.model_dim
        ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                        action_std=args.action_std,
                        lr=args.ppo_lr,
                        gamma=args.ppo_gamma,
                        K_epochs=args.K_epochs,
                        action_size=args.num_clusters)
    elif args.train_stage == 3:
        # if not specify the checkpoint path, use the default path produced from previous stage path.
        if args.checkpoint is None:
            args.checkpoint = str(Path(args.save_dir).parent / 'stage_2' / 'model_best.pth.tar')
        assert Path(args.checkpoint).exists(), f'{str(args.checkpoint)} is not exists!'

        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        fc.load_state_dict(checkpoint['fc'])

        state_dim = args.model_dim
        ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                        action_std=args.action_std,
                        lr=args.ppo_lr,
                        gamma=args.ppo_gamma,
                        K_epochs=args.K_epochs,
                        action_size=args.num_clusters)
        ppo.policy.load_state_dict(checkpoint['policy'])
        ppo.policy_old.load_state_dict(checkpoint['policy'])
    else:
        raise ValueError

    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    fc = fc.cuda()

    assert model is not None, "creating model failed. "
    print(model)
    print(f"model Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"fc Total params: {sum(p.numel() for p in fc.parameters()) / 1e6:.2f}M")
    return model, fc, ppo


def get_optimizer(args, model, fc):
    if args.train_stage != 2:
        params = [{'params': model.parameters(), 'lr': args.backbone_lr},
                  {'params': fc.parameters(), 'lr': args.fc_lr}]
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params,
                                        lr=0,  # specify in params
                                        momentum=args.momentum,
                                        nesterov=args.nesterov,
                                        weight_decay=args.wdecay)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, betas=(args.beta1, args.beta2), weight_decay=args.wdecay) #0.9, 0.999, 1e-5
        else:
            raise NotImplementedError
    else:
        optimizer = None
        args.epochs = args.ppo_epochs
    return optimizer


def get_scheduler(args, optimizer):#学习率的调整策略
    if optimizer is None:
        return None
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup, eta_min=1e-6)#优化器， 最大迭代次数，最小学习率
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError
    return scheduler

# Basic Functions ------------------------------------------------------------------------------------------------------
def train_GCN(args, train_set, model, fc, ppo, criterion, optimizer, scheduler, tb_writer, save_dir):
    # Init variables of logging training process
    save_dir = Path(save_dir)
    best_train_loss = BestVariable(order='min')
    header = ['epoch', 'train', 'best_epoch', 'best_train']
    losses_csv = CSVWriter(filename=save_dir / 'losses.csv', header=header)
    results_csv = CSVWriter(filename=save_dir / 'results.csv', header=['epoch', 'final_epoch', 'final_loss'])
    early_stop = EarlyStop(max_num_accordance=args.patience) if args.patience is not None else None

    if args.train_stage == 2:  # stage-2 just training RL module
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()
    memory_list = [rlmil.Memory(), rlmil.Memory()]  # the memory of two branch
    for epoch in range(args.epochs):
        print(f"Training Stage: {args.train_stage}, lr:")
        if optimizer is not None:
            for k, group in enumerate(optimizer.param_groups):
                print(f"group[{k}]: {group['lr']}")

        train_set.shuffle()
        length = len(train_set)

        losses = [AverageMeter() for _ in range(args.T)]
        reward_list = [AverageMeter() for _ in range(args.T - 1)]

        progress_bar = tqdm(range(args.num_data))
        feat_list, cluster_list, step, coord_list = [], [], 0, []
        batch_idx = 0

        for data_idx in progress_bar:
            HGNN_data = {}
            loss_list = []

            feat, cluster, label, case_id, coord = train_set[data_idx % length]
            # HGNN_data['n'] = feat.shape[-2]
            # HGNN_data['hypergraph'] = {str(j): cluster[j] for j in range(len(cluster))}
            # HGNN_list.append(HGNN_data)
            # print(f"before train, HGNN_data:{HGNN_data}")
            # 主要初始化features和label
            # H_GNN.data_initialise(HGNN_data, args)
            # A WSI features is a tensor of shape (num_patches, dim_features)
            assert len(feat.shape) == 2, f"{feat.shape}"
            feat = feat.unsqueeze(0).to(args.device)
            # coord = coord.unsqueeze(0).to(args.device)

            feat_list.append(feat)
            coord_list.append(coord)
            cluster_list.append(cluster)

            step += 1
            if step == args.batch_size:
                # first, random choice
                action_sequences = [torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device) for _ in
                                    range(2)]   #because two branches for contrastive learning
                # x_views = [get_feats(feat_list, cluster_list, action_sequence=a, feat_size=args.feat_size) for a in
                #            action_sequences]  # get WSI-Fset
                graph_views = []
                for a in action_sequences:
                    x_view, construction_list, coord_distribution_list = get_feats(feat_list, cluster_list, coord_list, action_sequence=a, feat_size=args.feat_size, feat_size_ratio=args.feat_size_ratio, coord_clusters=args.coord_clusters)  #get WSI-Fset
                    # hgs = [dhg.Hypergraph(args.feat_size, construction + coord_distribution_list[i]) for i, construction in enumerate(construction_list)]
                    hgs = [dhg.Hypergraph(args.feat_size, construction) for i, construction in
                           enumerate(construction_list)]
                    # print(hgs[0].H.shape)
                    # assert 1==0, f""
                    x_view = mixup(x_view, args.alpha)[0]
                    graph_views.append((x_view, hgs))

                if args.train_stage != 2:
                    # aggregate patch-level features to WSI-level vector
                    outputs, states = model(graph_views)  # outputs is the same, but states have been detached
                    outputs = [fc(o, restart=True) for o in outputs]
                else:  # stage 2 just training R
                    with torch.no_grad():
                        outputs, states = model(graph_views)
                        outputs = [fc(o, restart=True) for o in outputs]
                assert len(outputs) == 2, f""
                loss = criterion(outputs[0], outputs[1])
                loss_list.append(loss)
                losses[0].update(loss.data.item(), len(feat_list))

                similarity_last = torch.cosine_similarity(outputs[0], outputs[1]).view(1, -1)
                for patch_step in range(1, args.T):
                    # select features by R(ppo)
                    if args.train_stage == 1:  # stage 1 doesn't have module ppo
                        action_sequences = [torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
                                            for _ in range(2)]
                    else:
                        if patch_step == 1:
                            # create indices by different states and memory for two views
                            action_sequences = [ppo.select_action(s.to(0), m, restart_batch=True) for s, m in
                                                zip(states, memory_list)]
                        else:
                            action_sequences = [ppo.select_action(s.to(0), m) for s, m in zip(states, memory_list)]

                    graph_views = []
                    for a in action_sequences:
                        x_view, construction_list, coord_distribution_list = get_feats(feat_list, cluster_list, coord_list, action_sequence=a,
                                                              feat_size=args.feat_size,
                                                              feat_size_ratio=args.feat_size_ratio)  # get WSI-Fset
                        # hgs = [dhg.Hypergraph(args.feat_size, construction + coord_distribution_list[i]) for i, construction in enumerate(construction_list)]
                        hgs = [dhg.Hypergraph(args.feat_size, construction) for i, construction in
                               enumerate(construction_list)]
                        x_view = mixup(x_view, args.alpha)[0]
                        # G = model.encoder.update_graph(construction_list, coord_distribution_list, args.feat_size, args)
                        graph_views.append((x_view, hgs))
                    if args.train_stage != 2:
                        outputs, states = model(graph_views)
                        outputs = [fc(o, restart=False) for o in outputs]
                    else:
                        with torch.no_grad():
                            outputs, states = model(graph_views)
                            outputs = [fc(o, restart=False) for o in outputs]

                    loss = criterion(outputs[0], outputs[1])
                    loss_list.append(loss)
                    losses[patch_step].update(loss.data.item(), len(feat_list))

                    similarity = torch.cosine_similarity(outputs[0], outputs[1]).view(1, -1)
                    reward = similarity_last - similarity  # decrease similarity for reward
                    similarity_last = similarity

                    reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                    for m in memory_list:
                        m.rewards.append(reward)

                # Update models parameters
                loss = sum(loss_list) / args.T
                if args.train_stage != 2:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    for m in memory_list:
                        ppo.update(m)

                # clean temp batch variables
                for m in memory_list:
                    m.clear_memory()
                feat_list, cluster_list, step, coord_list = [], [], 0, []
                batch_idx += 1

                progress_bar.set_description(
                    f"Train Epoch: {epoch + 1:2}/{args.epochs:2}. Iter: {batch_idx:3}/{args.eval_step:3}. "
                    f"Loss: {losses[-1].avg:.4f}. "
                )
                progress_bar.update()
        progress_bar.close()
        if scheduler is not None and epoch >= args.warmup:
            scheduler.step()

        train_loss = losses[-1].avg
        # Write to tensorboard
        if tb_writer is not None:
            tb_writer.add_scalar('train/1.train_loss', train_loss, epoch)

        # Choose the best result
        is_best = best_train_loss.compare(train_loss, epoch + 1, inplace=True)
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'fc': fc.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
            'policy': ppo.policy.state_dict() if ppo else None,
        }
        save_checkpoint(state, is_best, str(save_dir))
        # Logging
        losses_csv.write_row([epoch + 1, train_loss, best_train_loss.epoch, best_train_loss.best])
        results_csv.write_row([epoch + 1, best_train_loss.epoch, best_train_loss.best])
        print(f"Loss: {train_loss:.4f}, Best: {best_train_loss.best:.4f}, Epoch: {best_train_loss.epoch:2}\n")

        # Early Stop
        if early_stop is not None:
            early_stop.update(best_train_loss.best)
            if early_stop.is_stop():
                break

    if tb_writer is not None:
        tb_writer.close()


def run(args):
    init_seeds(args.seed)

    if args.save_dir is None:
        create_save_dir(args)
    else:
        args.save_dir = str(Path(args.base_save_dir) / args.save_dir)
    args.save_dir = increment_path(Path(args.save_dir), exist_ok=args.exist_ok, sep='_')  # increment run
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    # Dataset
    train_set, dim_patch, train_length = get_datasets(args)
    args.num_data = train_length * args.data_repeat  #对比学习需要更多的数据
    args.eval_step = int(args.num_data / args.batch_size)
    # if args.arch == "HGNN":
    #     HGNN_data = {}
    #     feat, cluster, label, case_id = train_set[0]
    #     assert len(feat.shape) == 2, f"{feat.shape}"
    #     # feat = feat.squeeze(0)
    #     label = label.unsqueeze(0)
    #
    #     HGNN_data['n'] = feat.shape[-2]
    #     HGNN_data['hypergraph'] = {str(j): cluster[j] for j in range(len(cluster))}
    #     HGNN_data['features'] = feat
    #     HGNN_data['labels'] = label
    #     args.GCN_hidden_dim = dim_patch
    # else:
    #     HGNN_data = None
    print(f"train_length: {train_length}, epoch_step: {args.num_data}, eval_step: {args.eval_step}")
    # assert train_set is None, "train_set已保存"

    # Model, Criterion, Optimizer and Scheduler
    model, fc, ppo = create_model(args, dim_patch)
    criterion = NT_Xent(args.batch_size, args.temperature)  # 损失函数, 128, 1.0
    optimizer = get_optimizer(args, model, fc)
    scheduler = get_scheduler(args, optimizer)

    # Save arguments
    with open(Path(args.save_dir) / 'args.yaml', 'w') as fp:
        yaml.dump(vars(args), fp, sort_keys=False)
    print(args, '\n')

    # TensorBoard
    tb_writer = SummaryWriter(args.save_dir) if args.use_tensorboard else None

    # Start training
    TRAIN[args.arch](args, train_set, model, fc, ppo, criterion, optimizer, scheduler, tb_writer, args.save_dir)


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='local_data',
                        help="dataset name")
    parser.add_argument('--data_csv', type=str, default='',
                        help="the .csv filepath used")
    parser.add_argument('--data_split_json', type=str, default='/path/to/data_split.json')
    parser.add_argument('--preload', action='store_true', default=False,
                        help="preload the patch features, default False")
    parser.add_argument('--data_repeat', type=int, default=1,
                        help="contrastive learning need more iteration to train, the arg is to repeat data training for one epoch")
    parser.add_argument('--feat_size_ratio', default=0.5, type=float,
                        help="the size of selected WSI set. (we recommend 1024 at 20x magnification")
    parser.add_argument('--feat_size', default=1024, type=int,
                        help="the size of selected WSI set. (we recommend 1024 at 20x magnification")
    # Train
    parser.add_argument('--train_stage', default=1, type=int,
                        help="select training stage \
                              stage-1 : warm-up \
                              stage-2 : learn to select patches with RL \
                              stage-3 : finetune")
    parser.add_argument('--T', default=6, type=int,
                        help="maximum length of the sequence of RNNs")
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'],
                        help="specify the optimizer used, default Adam")
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'StepLR', 'CosineAnnealingLR'],
                        help="specify the lr scheduler used, default None")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="the batch size for training")
    parser.add_argument('--epochs', type=int, default=100,
                        help="")
    parser.add_argument('--ppo_epochs', type=int, default=40,
                        help="the training epochs for R")
    parser.add_argument('--backbone_lr', default=1e-4, type=float,
                        help='the learning rate for MIL encoder')
    parser.add_argument('--fc_lr', default=1e-4, type=float,
                        help='the learning rate for FC')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="the temperature coefficient of contrastive loss")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="the momentum of SGD optimizer")
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--warmup', default=0, type=float,
                        help="the number of epoch for training without lr scheduler, if scheduler is not None")
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help="the weight decay of optimizer")
    parser.add_argument('--patience', type=int, default=None,
                        help="if the loss not change during `patience` epochs, the training will early stop")

    # Architecture
    parser.add_argument('--checkpoint', default=None, type=str,
                        help="path to the stage-1/2 checkpoint (for training stage-2/3)")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--arch', default='CLAM_SB', type=str, choices=MODELS, help='model name')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--model_dim', type=int, default=512)
    # Architecture - PPO
    parser.add_argument('--policy_hidden_dim', type=int, default=512)
    parser.add_argument('--policy_conv', action='store_true', default=False)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=0.00001)
    parser.add_argument('--ppo_gamma', type=float, default=0.1)
    parser.add_argument('--K_epochs', type=int, default=3)
    # Architecture - Full_layer
    parser.add_argument('--feature_num', type=int, default=512)
    parser.add_argument('--fc_hidden_dim', type=int, default=1024)
    parser.add_argument('--fc_rnn', action='store_true', default=True)
    # Architecture - ABMIL
    parser.add_argument('--D', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    # CLAM
    parser.add_argument('--size_arg', type=str, default='small', choices=['small', 'big'])
    parser.add_argument('--k_sample', type=int, default=8)
    # HGNN
    parser.add_argument('--mediators', type=bool, default=True,
                        help='True for Laplacian with mediators, False for Laplacian without mediators')
    parser.add_argument('--fast', type=bool, default=False, help='faster version of HGCN (True)')
    # parser.add_agrument('--split', type=int, default=split, help='train-test split used for the dataset')
    parser.add_argument('--GCN_depth', type=int, default=2, help='number of hidden layers')
    parser.add_argument('--GCN_hidden_dim', type=int, default=1024, help='hidden dim of hidden layers, patch_dim/2')
    parser.add_argument('--GCN_dropout', type=float, default=0.5, help='dropout probability for GCN hidden layer')
    parser.add_argument('--GCN_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--GCN_decay', type=float, default=0.005, help='weight decay')
    parser.add_argument('--GCN_epochs', type=int, default=40)
    parser.add_argument('--Cluster', type=int, default=10, help='number of patch feature cluster')
    parser.add_argument('--coord_clusters', type=int, default=5, help='number of patch feature cluster')
    # Loss
    parser.add_argument('--use_tensorboard', action='store_true', default=False)
    # Save
    parser.add_argument('--base_save_dir', type=str, default='../results/HCC_OS_new_cross')
    parser.add_argument('--save_dir', type=str, default=None,
                        help="specify the save directory to save experiment results, default None."
                             "If not specify, the directory will be create by function create_save_dir(args)")
    parser.add_argument('--save_dir_flag', type=str, default=None,
                        help="append a `string` to the end of save_dir")
    parser.add_argument('--exist_ok', action='store_true', default=False)
    # Global
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help="random state")
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    # Pandas print setting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    torch.set_num_threads(1)

    # Global variables
    MODELS = ['ABMIL', 'CLAM_SB', 'HGNN']
    TRAIN = {
        'HGNN': train_GCN,
    }
    main()
