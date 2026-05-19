import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict

from metrics import compute_dice_score
from prompt_strategies import PromptSelector


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator if shuffle else None
    )


def preprocess_batch(images, prompts, processor, device, config):
    numpy_images = []
    for img in images:
        if hasattr(img, 'cpu'):
            numpy_img = img.cpu().numpy().transpose(1, 2, 0)
        else:
            numpy_img = np.array(img).transpose(1, 2, 0) if len(np.array(img).shape) == 3 else np.array(img)
        numpy_images.append(np.clip(numpy_img, 0.0, 1.0))

    image_inputs = processor.image_processor(images=numpy_images, return_tensors="pt", do_rescale=False)
    text_inputs = processor.tokenizer(
        text=prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=config['text_max_length']
    )
    inputs = {**image_inputs, **text_inputs}
    for k in inputs:
        if torch.is_tensor(inputs[k]):
            inputs[k] = inputs[k].to(device, non_blocking=True)
    return inputs


class MedVLSMDataset(Dataset):

    def __init__(self, data_root, dataset_name, split, prompt_strategy, config):
        self.data_root = Path(data_root)
        self.dataset_path = self.data_root / dataset_name
        self.split = split
        self.prompt_selector = PromptSelector(prompt_strategy)
        self.config = config
        with open(self.dataset_path / 'anns' / f'{split}.json', 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = Image.open(self.dataset_path / 'images' / ann['img_name']).convert('RGB')
        mask = Image.open(self.dataset_path / 'masks' / ann['mask_name']).convert('L')
        prompt = self.prompt_selector.select_prompt(ann)

        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(np.array(mask)).float()
        sz = self.config['image_size']
        img_t = F.interpolate(img_t.unsqueeze(0), size=(sz, sz), mode='bilinear')[0]
        mask_t = F.interpolate(mask_t.unsqueeze(0).unsqueeze(0), size=(sz, sz), mode='nearest')[0, 0]

        return {
            'image': torch.clamp(img_t, 0.0, 1.0),
            'mask': (mask_t > 0.5).long(),
            'prompt': prompt,
            'image_path': str(self.dataset_path / 'images' / ann['img_name']),
            'img_name': ann['img_name']
        }


def save_model_checkpoint(model, processor, save_path, task_info=None, config=None):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model.clipseg_backbone.save_pretrained(save_path / "clipseg_backbone")
    processor.save_pretrained(save_path / "processor")
    torch.save(model.state_dict(), save_path / "peft_model.pth")

    ewc_state = {
        'modality_fisher': {
            mid: {k: v.cpu() for k, v in fisher.items()}
            for mid, fisher in model.ewc_manager.modality_fisher.items()
        },
        'modality_params': {
            mid: {k: v.cpu() for k, v in params.items()}
            for mid, params in model.ewc_manager.modality_params.items()
        },
        'modality_task_count': model.ewc_manager.modality_task_count,
        'task_modality_id': model.ewc_manager.task_modality_id,
    }
    torch.save(ewc_state, save_path / "ewc_state.pth")

    modality_state = {
        'task_modality_id': dict(model.modality_manager.task_modality_id),
        'total_tasks': model.modality_manager.total_tasks,
        'alpha': model.modality_manager.alpha,
        'modality_prompts': dict(model.modality_manager.modality_prompts),
        'modality_stats': {
            mid: {'n': s['n'], 'mean': s['mean'].cpu()}
            for mid, s in model.modality_manager.modality_stats.items()
        },
        'intra_sim_stats': model.modality_manager.intra_sim_stats,
        'inter_sim_stats': model.modality_manager.inter_sim_stats,
        'current_task_id': model.current_task_id,
    }
    torch.save(modality_state, save_path / "modality_state.pth")

    with open(save_path / "checkpoint_info.json", 'w') as f:
        json.dump({
            "model_type": "CLIPModalityLoRAModel",
            "task_info": task_info,
            "config": config
        }, f, indent=2, default=str)
    return save_path


def load_model_full(save_path, config):
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    from model import CLIPModalityLoRAModel

    save_path = Path(save_path)
    
    processor_path = save_path / "processor"
    backbone_path = save_path / "clipseg_backbone"
    fallback = "CIDAS/clipseg-rd64-refined"
    
    processor = CLIPSegProcessor.from_pretrained(
        processor_path if processor_path.exists() else fallback,
        local_files_only=processor_path.exists()
    )
    backbone = CLIPSegForImageSegmentation.from_pretrained(
        backbone_path if backbone_path.exists() else fallback,
        local_files_only=backbone_path.exists()
    )
    model = CLIPModalityLoRAModel(backbone, processor, config)

    mod_path = save_path / "modality_state.pth"
    if mod_path.exists():
        mod_state = torch.load(mod_path, map_location=config['device'])
        model.modality_manager.task_modality_id = mod_state['task_modality_id']
        model.modality_manager.total_tasks = mod_state['total_tasks']
        model.modality_manager.alpha = mod_state['alpha']
        model.modality_manager.modality_prompts = mod_state['modality_prompts']
        model.modality_manager.modality_stats = {
            mid: {'n': s['n'], 'mean': s['mean'].to(config['device'])}
            for mid, s in mod_state['modality_stats'].items()
        }
        model.modality_manager.intra_sim_stats = mod_state['intra_sim_stats']
        model.modality_manager.inter_sim_stats = mod_state['inter_sim_stats']
        model.current_task_id = mod_state['current_task_id']


    for tid, mod_id in model.modality_manager.task_modality_id.items():
        tid = int(tid) if isinstance(tid, str) else tid
        model.set_modality(mod_id)
        first_task = model.modality_manager.get_first_task_in_modality(mod_id)
        if first_task is not None and first_task != tid:
            model.add_task_components(tid, reuse_from=first_task)
        else:
            model.add_task_components(tid)

    state_dict = torch.load(save_path / "peft_model.pth", map_location=config['device'])
    model.load_state_dict(state_dict, strict=False)

    ewc_path = save_path / "ewc_state.pth"
    if ewc_path.exists():
        ewc_state = torch.load(ewc_path, map_location=config['device'])
        model.ewc_manager.modality_fisher = {
            mid: {k: v.to(config['device']) for k, v in fisher.items()}
            for mid, fisher in ewc_state['modality_fisher'].items()
        }
        model.ewc_manager.modality_params = {
            mid: {k: v.to(config['device']) for k, v in params.items()}
            for mid, params in ewc_state['modality_params'].items()
        }
        model.ewc_manager.modality_task_count = ewc_state['modality_task_count']
        model.ewc_manager.task_modality_id = ewc_state['task_modality_id']

    return model, processor


def evaluate_task(model, processor, data_root, dataset_name, device, config, prompt_strategy, task_id):
    dataset = MedVLSMDataset(data_root, dataset_name, 'val', prompt_strategy, config)
    loader = get_dataloader(dataset, config['eval_batch_size'], shuffle=False,
                           num_workers=config['num_workers'], seed=42)

    modality_id = model.modality_manager.task_modality_id.get(task_id, 0)
    model.set_modality(modality_id)
    model.eval()

    total_dice, total_loss, n, nb = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True).long()
            inputs = preprocess_batch(images, batch['prompt'], processor, device, config)
            outputs = model(inputs, prompts=batch['prompt'], task_id=task_id)

            logits = outputs['logits']
            total_loss += F.cross_entropy(logits, masks).item()
            nb += 1

            preds = torch.argmax(logits, dim=1)
            for i in range(preds.shape[0]):
                total_dice += compute_dice_score(preds[i], masks[i])
                n += 1

    return total_dice / max(n, 1), total_loss / max(nb, 1)


def train_step(model, batch, optimizer, config, task_id):
    device = config['device']
    images = batch['image'].to(device, non_blocking=True)
    masks = batch['mask'].to(device, non_blocking=True).long()
    prompts = batch['prompt']

    inputs = preprocess_batch(images, prompts, model.processor, device, config)
    outputs = model(inputs, prompts=prompts, task_id=task_id)
    logits = outputs['logits']

    seg_loss = F.cross_entropy(logits, masks)

    preds_soft = F.softmax(logits, dim=1)[:, 1]
    true_mask = (masks == 1).float()
    intersection = (preds_soft * true_mask).sum(dim=(1, 2))
    union = preds_soft.sum(dim=(1, 2)) + true_mask.sum(dim=(1, 2))
    dice_loss = (1 - 2.0 * intersection / (union + 1e-8)).mean()

    ewc_loss = model.get_ewc_loss()

    total = (config['loss_weights']['segmentation'] * seg_loss +
             config['loss_weights']['dice'] * dice_loss +
             config['loss_weights']['ewc'] * ewc_loss)

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), config['gradient_clip_norm'])
    optimizer.step()

    return {
        'total': total.item(),
        'seg': seg_loss.item(),
        'dice': dice_loss.item(),
        'ewc': ewc_loss.item() if torch.is_tensor(ewc_loss) else ewc_loss
    }


def serialize_config(config):
    serializable = {}
    for k, v in config.items():
        if k == 'device':
            serializable[k] = str(v)
        elif isinstance(v, torch.dtype):
            serializable[k] = str(v)
        elif isinstance(v, (int, float, str, bool, list, dict)):
            serializable[k] = v
        else:
            serializable[k] = str(v)
    return serializable


def train_model(model, processor, config, datasets, order_name):
    print(f"\n{'='*60}\nTraining: {order_name.upper()} (Adaptive CRP + LoRA + EWC)\n{'='*60}")

    performance_matrix = {}
    task_results = {}
    checkpoints_dir = Path(config['output_dir']) / 'checkpoints' / order_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    prompt_strategy = config.get('prompt_strategy', 'concise')
    best_task_performance = {}

    for task_idx, ds_info in enumerate(datasets):
        print(f"\n--- Task {task_idx + 1}/{len(datasets)}: {ds_info['name']} ---")

        train_ds = MedVLSMDataset(config['data_root'], ds_info['name'], 'train', prompt_strategy, config)
        train_loader = get_dataloader(train_ds, config['batch_size'], shuffle=True,
                                      num_workers=config['num_workers'], seed=42)

        task_id, modality_id = model.start_new_task(
            train_ds[0]['prompt'], ds_info['name'], train_loader
        )

        optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(modality_id),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['max_epochs_per_task']
        )

        task_losses, val_dices = [], []
        best_dice, patience_cnt = 0.0, 0

        for epoch in range(config['max_epochs_per_task']):
            model.train()
            losses = {'total': 0, 'seg': 0, 'dice': 0, 'ewc': 0}
            for batch in train_loader:
                step_losses = train_step(model, batch, optimizer, config, task_id)
                for k in losses:
                    losses[k] += step_losses[k]
            for k in losses:
                losses[k] /= len(train_loader)
            task_losses.append(losses['total'])
            scheduler.step()

            val_dice, val_loss = evaluate_task(
                model, processor, config['data_root'], ds_info['name'],
                config['device'], config, prompt_strategy, task_id
            )
            val_dices.append(val_dice)

            if (epoch + 1) % config['print_interval'] == 0 or epoch == 0:
                print(f"  E{epoch+1}: loss={losses['total']:.4f} | val={val_dice:.4f}")

            if val_dice > best_dice + config['min_delta']:
                best_dice, patience_cnt = val_dice, 0
            else:
                patience_cnt += 1

            if epoch >= config['min_epochs'] and patience_cnt >= config['patience']:
                print(f"  Early stop at epoch {epoch+1}")
                break

        model.finish_task(train_loader, config['device'], task_id, modality_id)

        task_results[ds_info['name']] = {
            'final_loss': task_losses[-1] if task_losses else float('inf'),
            'val_dices': val_dices,
            'best_val_dice': best_dice,
            'total_epochs': len(task_losses),
            'modality': modality_id
        }

        print(f"\nAll tasks evaluation:")
        perf = {}
        for eval_tid in range(task_idx + 1):
            dice, _ = evaluate_task(
                model, processor, config['data_root'], datasets[eval_tid]['name'],
                config['device'], config, prompt_strategy, eval_tid
            )
            task_name = datasets[eval_tid]['name']
            task_modality = model.modality_manager.task_modality_id.get(eval_tid, -1)
            perf[task_name] = dice
            print(f"  {task_name} (M{task_modality}): {dice:.4f}")

            if task_name not in best_task_performance:
                best_task_performance[task_name] = dice
            else:
                best_task_performance[task_name] = max(best_task_performance[task_name], dice)

        if task_idx > 0:
            forgetting_values = []
            for i in range(task_idx):
                task_name = datasets[i]['name']
                forgetting_values.append(max(0, best_task_performance[task_name] - perf[task_name]))
            print(f"  Forgetting: {np.mean(forgetting_values):.4f}")

        performance_matrix[f"after_task_{task_idx+1}"] = perf

    # Final evaluation
    analysis = model.get_analysis_info()
    modality_info = analysis['modality_analysis']

    final_perfs = {}
    print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")

    modality_tasks = defaultdict(list)
    for i, ds in enumerate(datasets):
        task_modality = model.modality_manager.task_modality_id.get(i, -1)
        modality_tasks[task_modality].append((i, ds['name']))

    print("\n--- Results by Modality ---")
    for mid in sorted(modality_tasks.keys()):
        for tid, task_name in modality_tasks[mid]:
            dice, loss = evaluate_task(
                model, processor, config['data_root'], task_name,
                config['device'], config, prompt_strategy, tid
            )
            final_perfs[task_name] = {'dice_score': dice, 'loss': loss, 'modality': mid}

        dices = [final_perfs[name]['dice_score'] for _, name in modality_tasks[mid]]
        names = [f"{name}: {final_perfs[name]['dice_score']:.4f}" for _, name in modality_tasks[mid]]
        print(f"  M{mid} ({len(dices)} tasks, avg={np.mean(dices):.4f}): {', '.join(names)}")

    avg_dice = np.mean([p['dice_score'] for p in final_perfs.values()])
    first_perfs = [performance_matrix[f"after_task_{i+1}"].get(datasets[i]['name'], 0)
                   for i in range(len(datasets))]
    final_list = [final_perfs[datasets[i]['name']]['dice_score'] for i in range(len(datasets))]
    avg_forgetting = np.mean([max(0, f - l) for f, l in zip(first_perfs, final_list)])

    print(f"\n--- Summary ---")
    print(f"  Average Dice: {avg_dice:.4f}")
    print(f"  Average Forgetting: {avg_forgetting:.4f}")
    print(f"  Modalities: {modality_info['num_modalities']}, "
          f"Trainable params: {analysis['trainable_params']:,}")


    if config.get('save_checkpoints', True):
        final_checkpoint_path = checkpoints_dir / "final_model"
        final_checkpoint_info = {
            "order": order_name,
            "num_tasks": len(datasets),
            "tasks_trained": [d['name'] for d in datasets],
            "final_performances": final_perfs,
            "average_dice": avg_dice,
            "average_forgetting": avg_forgetting,
            "performance_matrix": performance_matrix,
            "method": "Adaptive-CRP + LoRA + EWC",
            "crp_alpha": model.modality_manager.alpha,
            "trainable_params": analysis['trainable_params'],
            "total_params": analysis['total_params'],
            "modality_analysis": modality_info
        }
        save_model_checkpoint(model, processor, final_checkpoint_path,
                            final_checkpoint_info, serialize_config(config))
        print(f"\nFinal model saved: {final_checkpoint_path}")

    results_dir = Path(config['output_dir']) / order_name
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'modality_analysis.json', 'w') as f:
        json.dump(modality_info, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    return performance_matrix, task_results, avg_forgetting, analysis