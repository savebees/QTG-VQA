import os, sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import logging
from termcolor import colored
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from torch.utils.data.distributed import DistributedSampler
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from DataLoader import VideoQADataLoader
from utils import todevice
from validate import validate
import model.TempAligner as TempAligner

from utils import todevice
from config import cfg, cfg_from_file
import clip
import random
import time 
from SemanticAligner import SemanticAligner

def train(cfg, args):
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
        'annotation_file': cfg.dataset.annotation_file,
        'appearance_feat': cfg.dataset.appearance_feat,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'pin_memory': True,
        'shuffle': True
    }
    train_loader = VideoQADataLoader(**train_loader_kwargs)

    logging.info("number of train instances: {}".format(len(train_loader.dataset)))
    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device) 
    clip_model.float()
    model_kwargs = {
        'vision_dim': cfg.train.vision_dim,
        'module_dim': cfg.train.module_dim,
    }
    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab'}
    tempaligner = TempAligner.TempAligner(**model_kwargs).to(device)

    semanticaligner = SemanticAligner().to(device)

    optimizer = optim.Adam(
    [
        {"params": tempaligner.parameters(), 'lr': cfg.train.lr},
        {"params": semanticaligner.parameters(), 'lr': cfg.train.lr},
    ]
    )

    mseloss = nn.MSELoss()
    start_epoch = 0
    best_val = 0
    if cfg.train.restore:
        print("Restore checkpoint and optimizer...")
        ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
    logging.info("Start training........")
    # 归一化的频率倒数作为权重
    q_type_freq = {'U': 35178, 'A': 11094, 'C': 2824, 'R': 2470, 'I': 2380, 'F': 2514}
    total_count = sum(q_type_freq.values())
    q_type_weights = {q_type: (total_count / freq) / total_count for q_type, freq in q_type_freq.items()}

    for epoch in range(start_epoch, cfg.train.max_epochs):
        logging.info('>>>>>> epoch {epoch} <<<<<<'.format(epoch=colored("{}".format(epoch), "green", attrs=["bold"])))
        tempaligner.train()
        semanticaligner.train()
        total_acc, count = 0, 0
        batch_mse_sum = 0.0
        total_loss, avg_loss = 0.0, 0.0
        avg_loss = 0
        total_ce_loss = 0.0
        avg_ce_loss = 0.0
        total_recon_loss = 0.0
        avg_recon_loss = 0.0
        train_accuracy = 0
        q_type_mapping = {'U': 0, 'A': 1, 'F': 2, 'R': 3, 'C': 4, 'I': 5}
        epoch_q_type_losses = {q_type: 0.0 for q_type in q_type_mapping.keys()}
        epoch_q_type_counts = {q_type: 0 for q_type in q_type_mapping.keys()}
        epoch_q_type_weighted_losses = {q_type: 0.0 for q_type in q_type_mapping.keys()}

        for i, batch in enumerate(iter(train_loader)):
            progress = epoch + i / len(train_loader)
            _, _, answers, ans_candidates, batch_clips_data, question, q_types = batch
            q_types = torch.tensor([q_type_mapping.get(q, -1) for q in q_types], dtype=torch.long).to(device)  # 使用get确保没有未映射项
            answers, ans_candidates, batch_clips_data, question = \
                answers.to(device), ans_candidates.to(device), batch_clips_data.to(device), question.to(device)

            batch_size = batch_clips_data.shape[0]
            feat_dim = batch_clips_data.shape[-1]
            num_ans = ans_candidates.shape[1] 
            ans_candidates = ans_candidates.view(-1, 77)
            with torch.no_grad():
                answers_features = clip_model.encode_text( ans_candidates ).float()
            answers_features = semanticaligner(answers_features, batch_clips_data).float()
            question_features = clip_model.encode_text( question.squeeze() ).float()
            video_appearance_feat = batch_clips_data.view(batch_size, -1, feat_dim) 
            answers_features = answers_features.view(batch_size, num_ans, -1)
            answers = answers.cuda().squeeze()
            batch_inputs = [answers,  answers_features, video_appearance_feat, question_features]
            optimizer.zero_grad()
            logits, visual_embedding_decoder = tempaligner(*batch_inputs)
            batch_agg = np.concatenate(np.tile(np.arange(batch_size).reshape([batch_size, 1]),
                                               [1, 4])) * 4  # [0, 0, 0, 0, 0, 5, 5, 5, 5, 1, ...]
            answers_agg = tile(answers, 0, 4)
            #if (q_types == -1).any():
            #   print("Warning: Unmapped q_type found!")  
            #print(f"Total q_types count in batch {i}: {len(q_types)}")  

            # 基于问题类型的损失计算
            q_type_losses = {q_type: torch.tensor(0.0, device=device) for q_type in q_type_mapping.keys()}
            q_type_counts = {q_type: 0 for q_type in q_type_mapping.keys()}

            for q_type, q_type_idx in q_type_mapping.items():
                q_type_tensor = torch.tensor([q_type_idx], dtype=torch.long, device=device)
                q_type_mask = (q_types == q_type_tensor).nonzero(as_tuple=True)[0]
                #print(f"Question Type {q_type}: Count in batch = {len(q_type_mask)}")

                if q_type_mask.numel() > 0:
                    q_type_indices = torch.cat([(q_type_mask * 4 + i).unsqueeze(1) for i in range(4)], dim=1).view(-1)
                    q_type_logits = logits[q_type_indices].view(-1, 4)   # [len(q_type_mask), 4]
                    q_type_correct_answer = answers[q_type_mask]
                    q_type_target_logits = torch.zeros_like(q_type_logits)
                    for i, correct_idx in enumerate(q_type_correct_answer):
                        correct_logit = q_type_logits[i, correct_idx]
                        q_type_target_logits[i] = correct_logit  # 广播机制创建 q_type_target_logits

                    # Debugging: Logging the logits and targets for each question type
                    #for idx in range(len(q_type_logits)):
                        #logging.info(f"Batch {i}, Question Type {q_type}, Index {idx}: Logit = {q_type_logits[idx].tolist()}, Target Logit = {q_type_target_logits[idx].tolist()}")
                    
                    q_type_loss_ce = torch.max(torch.zeros_like(q_type_logits), 1.0 + q_type_logits - q_type_target_logits)
                    q_type_loss_ce = q_type_loss_ce.sum()
                    qtype_weighted_loss = q_type_loss_ce * q_type_weights[q_type]                 
                    
                    q_type_losses[q_type] += q_type_loss_ce
                    q_type_counts[q_type] += q_type_mask.size(0)
                    epoch_q_type_weighted_losses[q_type] += qtype_weighted_loss.item()
            qtype_avg_loss = sum(q_type_losses.values()) / sum(q_type_counts.values())
            for q_type in q_type_losses.keys():
                epoch_q_type_losses[q_type] += q_type_losses[q_type]
                epoch_q_type_counts[q_type] += q_type_counts[q_type]
            total_weighted_loss = sum(epoch_q_type_weighted_losses.values())

            # Debugging: 打印logits和对应的目标logits
            #target_logits = logits[answers_agg + torch.from_numpy(batch_agg).cuda()]
            #for j in range(logits.size(0)):
                #logging.info(f"Batch {i}, Index {j}: Logits = {logits[j].detach().cpu().numpy()}, Target Logits = {target_logits[j].detach().cpu().numpy()}")

            loss_ce = torch.max(torch.tensor(0.0).cuda(),
                             1.0 + logits - logits[answers_agg + torch.from_numpy(batch_agg).cuda()])
            loss_ce = loss_ce.sum()
            recon_loss = mseloss(visual_embedding_decoder, video_appearance_feat)
            loss = 0.01*loss_ce + recon_loss + qtype_avg_loss + total_weighted_loss
            loss.backward()
            total_loss += loss.detach()
            avg_loss = total_loss / (i + 1)
            total_ce_loss += loss_ce.detach()
            avg_ce_loss = total_ce_loss / (i+1)
            total_recon_loss += recon_loss.detach()
            avg_recon_loss = total_recon_loss / (i+1)
            nn.utils.clip_grad_norm_(tempaligner.parameters(), max_norm=12)
            optimizer.step()
            preds = torch.argmax(logits.view(batch_size, 4), dim=1)
            aggreeings = (preds == answers)

            total_acc += aggreeings.sum().item()
            count += answers.size(0)
            train_accuracy = total_acc / count
            sys.stdout.write(
                "\rProgress = {progress}   sum_loss = {sum_loss}   avg_loss = {avg_loss}   ce_loss = {ce_loss}   recon_loss = {recon_loss}   avg_acc = {avg_acc}    exp: {exp_name}".format(
                    progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                    sum_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                    avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                    ce_loss=colored("{:.4f}".format(avg_ce_loss.item()), "red", attrs=['bold']),
                    recon_loss=colored("{:.4f}".format(avg_recon_loss.item()), "green", attrs=['bold']),
                    avg_acc=colored("{:.4f}".format(train_accuracy), "red", attrs=['bold']),
                    exp_name=cfg.exp_name))
            sys.stdout.flush()
        sys.stdout.write("\n")
        if (epoch + 1) % 10 == 0:
            optimizer = step_decay(cfg, optimizer)
        sys.stdout.flush()
        logging.info(
            "Epoch = {:d}, Sum Loss = {:.4f}, Avg Loss = {:.4f}, CE Loss = {:.4f}, Recon Loss = {:.4f}, Avg Acc = {:.4f}, Weighted Loss = {:.4f}, Qtype_avg Loss = {:.4f}".format(
                epoch, total_loss, avg_loss, avg_ce_loss, avg_recon_loss, train_accuracy, total_weighted_loss, qtype_avg_loss
            )
        )
        # Calculate average loss per question type for the epoch
        for q_type in q_type_mapping.keys():
            total_loss = epoch_q_type_losses[q_type]
            count = epoch_q_type_counts[q_type]
            avg_loss = total_loss / count if count > 0 else 0
            logging.info(f"Epoch {epoch}, Question Type {q_type}: Total Loss = {total_loss:.4f}, Count = {count}, Avg Loss = {avg_loss:.4f}")
        ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        else:
            assert os.path.isdir(ckpt_dir)
        save_checkpoint(epoch, tempaligner, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, 'tempaligner_{}.pt'.format(epoch)))
        save_checkpoint(epoch, semanticaligner, optimizer, None, os.path.join(ckpt_dir, 'semanticaligner_{}.pt'.format(epoch)))
        sys.stdout.write('\n >>>>>> save to %s <<<<<< \n' % ckpt_dir)
        sys.stdout.flush()



# Credit https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = (predicted == true)
    return agreeing


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
    }
    time.sleep(10)
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='sutd-traffic_transition.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    assert cfg.dataset.name in ['sutd-traffic']

    if not cfg.multi_gpus:
        torch.cuda.set_device(cfg.gpu_id)
    # make logging.info display into both shell and file
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)
    else:
        assert os.path.isdir(cfg.dataset.save_dir)
    log_file = os.path.join(cfg.dataset.save_dir, "log")
    if not cfg.train.restore and not os.path.exists(log_file):
        os.mkdir(log_file)
    else:
        assert os.path.isdir(log_file)

    fileHandler = logging.FileHandler(os.path.join(log_file, 'stdout.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(cfg).items():
        logging.info(k + ':' + str(v))
    # concat absolute path of input files

    if cfg.dataset.name == 'sutd-traffic':
        cfg.dataset.annotation_file = cfg.dataset.annotation_file.format('train')

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))

    else:
        pass

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train(cfg, args)


if __name__ == '__main__':
    main()

