import torch
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import json
import pickle
import logging
from termcolor import colored
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from DataLoader import VideoQADataLoader
from utils import todevice

import model.TempAligner as TempAligner

from config import cfg, cfg_from_file
import clip
from SemanticAligner import SemanticAligner

def validate(cfg, tempaligner, semanticaligner, clip_model, data, device, write_preds=False):
    tempaligner.eval()
    clip_model.eval()
    semanticaligner.eval()
    print('validating...')
    total_acc, count = 0.0, 0
    q_type_acc = {key: [0, 0] for key in ['U', 'A', 'F', 'R', 'C', 'I']}  
    all_preds, gts, v_ids, r_ids, q_types = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            record_id, video_idx, answers, ans_candidates, batch_clips_data, question, q_type = batch
            q_types = torch.tensor([q_type_mapping[q] for q in q_types], dtype=torch.long).to(device)
            record_id, video_idx, answers, ans_candidates, batch_clips_data, question = \
                record_id.to(device), video_idx.to(device), answers.to(device), ans_candidates.to(device), batch_clips_data.to(device), question.to(device)
            if cfg.train.batch_size == 1:
                answers = answers.to(device)
            else:
                answers = answers.to(device).squeeze()
            batch_size = answers.size(0)
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
            logits, visual_embedding_decoder = tempaligner(*batch_inputs)

            logits = logits.to(device)
            preds = torch.argmax(logits.view(batch_size, 4), dim=1)
            agreeings = (preds == answers)

            for idx, qt in enumerate(q_type):
                q_type_acc[qt][0] += agreeings[idx].item()
                q_type_acc[qt][1] += 1

            if write_preds:
                all_preds.extend(preds.tolist())
                gts.extend(answers.tolist())
                r_ids.extend(record_id.tolist())
                v_ids.extend(video_idx.tolist())
                q_types.extend(q_type.tolist())

            total_acc += agreeings.float().sum().item()
            count += answers.size(0)

        acc = total_acc / count
        logging.info(f'Validation set size: {count}')
        logging.info(f'Overall Accuracy on Validation set: {acc:.4f}')

        for qt in q_type_acc:
            if q_type_acc[qt][1] > 0:
                logging.info(f'Accuracy for {qt}: {q_type_acc[qt][0] / q_type_acc[qt][1]:.4f}')

    if not write_preds:
        return acc, q_type_acc
    else:
        return acc, q_type_acc, all_preds, gts, v_ids, r_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='sutd-traffic_transition.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    assert cfg.dataset.name in ['sutd-traffic']
    assert os.path.exists(cfg.dataset.data_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(cfg.dataset.save_dir, 'validation_log.txt'), filemode='w')

    best_acc = 0.0
    best_ckpt = {}
    q_types = ['U', 'A', 'F', 'R', 'C', 'I']
    for q_type in q_types:
        best_ckpt[q_type] = {'epoch': -1, 'accuracy': 0.0}
    checkpoint_dir = "./results/sutd-traffic/ckpt"
    cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))

    for epoch in range(50):  
        temp_ckpt = f"{checkpoint_dir}/tempaligner_{epoch}.pt"
        semantic_ckpt = f"{checkpoint_dir}/semanticaligner_{epoch}.pt"

        if os.path.exists(temp_ckpt) and os.path.exists(semantic_ckpt):
            print(f"Loading checkpoints from epoch {epoch}")

            loaded = torch.load(temp_ckpt, map_location=device)
            model_kwargs = loaded['model_kwargs']
            tempaligner =  TempAligner.TempAligner(**model_kwargs).to(device)
            model_dict = tempaligner.state_dict()
            state_dict = {k: v for k, v in loaded['state_dict'].items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            tempaligner.load_state_dict(model_dict)
            
            loaded_semantic = torch.load(semantic_ckpt, map_location=device)
            semanticaligner = SemanticAligner().to(device)
            semanticaligner.load_state_dict(loaded_semantic['state_dict'])

            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            clip_model.float()

            if cfg.dataset.name == 'sutd-traffic':
                cfg.dataset.annotation_file = cfg.dataset.annotation_file.format('test')
                test_loader_kwargs = {
                    'appearance_feat': cfg.dataset.appearance_feat,
                    'annotation_file': cfg.dataset.annotation_file,
                    'batch_size': cfg.train.batch_size,
                    'num_workers': cfg.num_workers,
                    'shuffle': False
                }
                test_loader = VideoQADataLoader(**test_loader_kwargs)

                acc, type_accs = validate(cfg, tempaligner, semanticaligner, clip_model, test_loader, device, False)
                logging.info(f'Epoch {epoch} Validation Accuracy: {acc:.4f}')
                if acc > best_acc:
                    best_acc = acc
                    best_ckpt['Overall'] = epoch
                for qt in q_types:
                    if type_accs[qt][1] > 0:
                        qt_acc = type_accs[qt][0] / type_accs[qt][1]
                        if qt_acc > best_ckpt[qt]['accuracy']:
                            best_ckpt[qt]['accuracy'] = qt_acc
                            best_ckpt[qt]['epoch'] = epoch


    for qt in q_types:
        logging.info(f'Best Accuracy for {qt}: {best_ckpt[qt]["accuracy"]:.4f} at epoch {best_ckpt[qt]["epoch"]}')
    logging.info(f'Best Overall Accuracy: {best_acc:.4f} at epoch {best_ckpt["Overall"]}')
    best_epoch_overall = best_ckpt['Overall']
    temp_ckpt = f"{checkpoint_dir}/tempaligner_{best_epoch_overall}.pt"
    semantic_ckpt = f"{checkpoint_dir}/semanticaligner_{best_epoch_overall}.pt"

    loaded = torch.load(temp_ckpt, map_location=device)
    model_kwargs = loaded['model_kwargs']
    tempaligner = TempAligner.TempAligner(**model_kwargs).to(device)
    model_dict = tempaligner.state_dict()
    state_dict = {k: v for k, v in loaded['state_dict'].items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    tempaligner.load_state_dict(model_dict)
    loaded_semantic = torch.load(semantic_ckpt, map_location=device)
    semanticaligner = SemanticAligner().to(device)
    semanticaligner.load_state_dict(loaded_semantic['state_dict'])
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.float()

    if cfg.test.write_preds:
        acc, all_preds, gts, v_ids, r_ids = validate(cfg, tempaligner, semanticaligner, clip_model, test_loader, device, True)
        results_file_path = os.path.join(cfg.dataset.save_dir, 'validation_results.jsonl')
        with open(results_file_path, 'w') as f:
            for pred, gt, vid_id, rec_id in zip(all_preds, gts, v_ids, r_ids):
                result = {
                    "video_id": vid_id,
                    "record_id": rec_id,
                    "predicted_answer": pred,
                    "true_answer": gt
                }
                json.dump(result, f)
                f.write('\n')
        print(f'Validation accuracy: {acc:.4f}')
        print(f'Results with prezdictions written to {results_file_path}')

    else:
        acc, type_accs = validate(cfg, tempaligner, semanticaligner, clip_model, test_loader, device, False)
        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()
