import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class AdaptiveCRPModalityManager:
    """
    Adaptive CRP modality manager using observed similarity distributions
    to set reference points for Bayesian modality assignment.
    """

    def __init__(self, clip_text_model, tokenizer, config, device):
        self.clip_text_model = clip_text_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.alpha = config.get('crp_alpha', 1.0)

        self.modality_stats = {}
        self.task_modality_id = {}
        self.modality_prompts = {}
        self.total_tasks = 0
        self.task_embeddings = {}

        # Online statistics for similarity distributions (Welford's algorithm)
        self.intra_sim_stats = {'n': 0, 'mean': 0.0, 'M2': 0.0}
        self.inter_sim_stats = {'n': 0, 'mean': 0.0, 'M2': 0.0}

    def extract_prompt_embedding(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]

        with torch.no_grad():
            inputs = self.tokenizer(
                prompts, return_tensors="pt",
                padding=True, truncation=True, max_length=77
            )
            for k in inputs:
                if torch.is_tensor(inputs[k]):
                    inputs[k] = inputs[k].to(self.device)

            outputs = self.clip_text_model(**inputs)
            embeddings = F.normalize(outputs.pooler_output, p=2, dim=-1)
            return embeddings.mean(dim=0)

    def extract_dataset_embedding(self, dataloader, num_samples=300):
        all_prompts = []
        sample_count = 0

        for batch in dataloader:
            prompts = batch['prompt']
            if isinstance(prompts, str):
                prompts = [prompts]
            all_prompts.extend(prompts)
            sample_count += len(prompts)
            if sample_count >= num_samples:
                break

        unique_prompts = list(set(all_prompts))
        embeddings = [self.extract_prompt_embedding(p) for p in unique_prompts]
        mean_embedding = F.normalize(torch.stack(embeddings).mean(dim=0), p=2, dim=-1)
        return mean_embedding, unique_prompts

    def _update_welford(self, stats, new_value):
        stats['n'] += 1
        delta = new_value - stats['mean']
        stats['mean'] += delta / stats['n']
        delta2 = new_value - stats['mean']
        stats['M2'] += delta * delta2

    def _get_std(self, stats):
        if stats['n'] < 2:
            return 0.1
        return np.sqrt(stats['M2'] / (stats['n'] - 1))

    def _update_similarity_distributions(self, task_embedding, assigned_modality):
        for mid, stats in self.modality_stats.items():
            sim = F.cosine_similarity(
                task_embedding.unsqueeze(0), stats['mean'].unsqueeze(0), dim=1
            ).item()

            if mid == assigned_modality:
                if stats['n'] > 1:
                    self._update_welford(self.intra_sim_stats, sim)
            else:
                self._update_welford(self.inter_sim_stats, sim)

    def _init_modality_stats(self, modality_id, embedding, prompts):
        self.modality_stats[modality_id] = {'n': 1, 'mean': embedding.clone()}
        self.modality_prompts[modality_id] = prompts.copy()

    def _update_modality_stats(self, modality_id, new_embedding, prompts):
        stats = self.modality_stats[modality_id]
        n = stats['n'] + 1
        stats['mean'] = stats['mean'] + (new_embedding - stats['mean']) / n
        stats['n'] = n

        for p in prompts:
            if p not in self.modality_prompts[modality_id]:
                self.modality_prompts[modality_id].append(p)

    def _compute_log_likelihood_ratio(self, sim):
        intra_mean = self.intra_sim_stats['mean']
        intra_std = max(self._get_std(self.intra_sim_stats), 0.05)
        inter_mean = self.inter_sim_stats['mean']
        inter_std = max(self._get_std(self.inter_sim_stats), 0.05)

        log_p_same = -0.5 * ((sim - intra_mean) / intra_std) ** 2 - np.log(intra_std)
        log_p_diff = -0.5 * ((sim - inter_mean) / inter_std) ** 2 - np.log(inter_std)
        return log_p_same - log_p_diff

    def assign_modality(self, task_id, dataloader):
        task_embedding, prompts = self.extract_dataset_embedding(dataloader)
        self.task_embeddings[task_id] = task_embedding.clone()

        if self.total_tasks == 0:
            modality_id = 0
            self._init_modality_stats(modality_id, task_embedding, prompts)
            self.task_modality_id[task_id] = modality_id
            self.total_tasks += 1
            return modality_id, 0.0, True, prompts[0] if prompts else "unknown"

        similarities = {}
        for mid, stats in self.modality_stats.items():
            sim = F.cosine_similarity(
                task_embedding.unsqueeze(0), stats['mean'].unsqueeze(0), dim=1
            ).item()
            similarities[mid] = sim

        max_sim = max(similarities.values())
        min_distance = 1.0 - max_sim

        log_probs = {}
        n_total = self.total_tasks
        has_distribution_info = (self.intra_sim_stats['n'] >= 1 and
                                 self.inter_sim_stats['n'] >= 1)

        for mid, stats in self.modality_stats.items():
            n_k = stats['n']
            s_t = similarities[mid]
            log_prior = np.log(n_k) - np.log(n_total + self.alpha)

            if has_distribution_info:
                log_likelihood = self._compute_log_likelihood_ratio(s_t)
            else:
                log_likelihood = np.log(s_t + 1e-6) - np.log(1 - s_t + 1e-6)

            log_probs[mid] = log_prior + log_likelihood

        log_prior_new = np.log(self.alpha) - np.log(n_total + self.alpha)
        if has_distribution_info:
            log_likelihood_new = -self._compute_log_likelihood_ratio(max_sim)
        else:
            novelty = 1 - max_sim
            log_likelihood_new = np.log(novelty + 1e-6) - np.log(1 - novelty + 1e-6)

        log_probs['new'] = log_prior_new + log_likelihood_new
        best_choice = max(log_probs, key=log_probs.get)

        self.total_tasks += 1

        if best_choice == 'new':
            modality_id = len(self.modality_stats)
            self._init_modality_stats(modality_id, task_embedding, prompts)
            self.task_modality_id[task_id] = modality_id
            self._update_similarity_distributions(task_embedding, modality_id)
            return modality_id, min_distance, True, prompts[0] if prompts else "unknown"
        else:
            modality_id = best_choice
            self._update_similarity_distributions(task_embedding, modality_id)
            self._update_modality_stats(modality_id, task_embedding, prompts)
            self.task_modality_id[task_id] = modality_id
            return modality_id, min_distance, False, self.modality_prompts[modality_id][0]

    def get_num_modalities(self):
        return len(self.modality_stats)

    def get_modality_task_count(self, modality_id):
        return self.modality_stats[modality_id]['n'] if modality_id in self.modality_stats else 0

    def get_first_task_in_modality(self, modality_id):
        for tid, mid in self.task_modality_id.items():
            if mid == modality_id:
                return tid
        return None

    def get_analysis_info(self):
        modality_info = {}
        for mid in self.modality_stats:
            tasks_in_modality = [tid for tid, m in self.task_modality_id.items() if m == mid]
            modality_info[mid] = {
                'task_count': self.modality_stats[mid]['n'],
                'tasks': tasks_in_modality,
                'representative_prompts': self.modality_prompts.get(mid, [])[:5],
            }

        return {
            'method': 'Adaptive-CRP + LoRA + EWC',
            'num_modalities': len(self.modality_stats),
            'crp_alpha': self.alpha,
            'total_tasks': self.total_tasks,
            'task_to_modality': self.task_modality_id,
            'modality_info': modality_info,
            'intra_sim_stats': {
                'mean': self.intra_sim_stats['mean'],
                'std': self._get_std(self.intra_sim_stats),
                'n': self.intra_sim_stats['n']
            },
            'inter_sim_stats': {
                'mean': self.inter_sim_stats['mean'],
                'std': self._get_std(self.inter_sim_stats),
                'n': self.inter_sim_stats['n']
            }
        }


class EWCManager:

    def __init__(self, lambda_base, fisher_samples, config):
        self.lambda_base = lambda_base
        self.fisher_samples = fisher_samples
        self.config = config
        self.modality_fisher = {}
        self.modality_params = {}
        self.modality_task_count = {}
        self.task_modality_id = {}

    def register_task(self, task_id, modality_id):
        self.task_modality_id[task_id] = modality_id
        if modality_id not in self.modality_task_count:
            self.modality_task_count[modality_id] = 0
        self.modality_task_count[modality_id] += 1

    def compute_fisher(self, model, dataloader, device, task_id, modality_id, config):
        from train import preprocess_batch

        trainable_params = model.get_trainable_parameters(modality_id)
        param_names = model.get_trainable_param_names(modality_id)
        fisher = {name: torch.zeros_like(p.data) for name, p in zip(param_names, trainable_params)}

        model.eval()
        sample_count = 0

        for batch in dataloader:
            if sample_count >= self.fisher_samples:
                break

            images = batch['image'].to(device)
            masks = batch['mask'].to(device).long()
            prompts = batch['prompt']

            inputs = preprocess_batch(images, prompts, model.processor, device, config)
            outputs = model(inputs, prompts=prompts, task_id=task_id)
            loss = F.cross_entropy(outputs['logits'], masks)

            model.zero_grad()
            loss.backward()

            for name, p in zip(param_names, trainable_params):
                if p.grad is not None:
                    fisher[name] += p.grad.data.pow(2) * len(images)
            sample_count += len(images)

        for name in fisher:
            fisher[name] /= max(sample_count, 1)

        return fisher, sample_count

    def consolidate(self, model, dataloader, device, task_id, modality_id, config):
        fisher, _ = self.compute_fisher(model, dataloader, device, task_id, modality_id, config)

        trainable_params = model.get_trainable_parameters(modality_id)
        param_names = model.get_trainable_param_names(modality_id)
        current_params = {name: p.data.clone() for name, p in zip(param_names, trainable_params)}

        if modality_id not in self.modality_fisher:
            self.modality_fisher[modality_id] = fisher
            self.modality_params[modality_id] = current_params
        else:
            alpha = 1.0 / self.modality_task_count[modality_id]
            for name in fisher:
                if name in self.modality_fisher[modality_id]:
                    self.modality_fisher[modality_id][name] = (
                        (1 - alpha) * self.modality_fisher[modality_id][name] +
                        alpha * fisher[name]
                    )
                    self.modality_params[modality_id][name] = current_params[name]
                else:
                    self.modality_fisher[modality_id][name] = fisher[name]
                    self.modality_params[modality_id][name] = current_params[name]

    def get_ewc_loss(self, model, current_modality_id):
        if current_modality_id not in self.modality_fisher:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        if self.modality_task_count.get(current_modality_id, 0) <= 1:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        fisher = self.modality_fisher[current_modality_id]
        old_params = self.modality_params[current_modality_id]

        trainable_params = model.get_trainable_parameters(current_modality_id)
        param_names = model.get_trainable_param_names(current_modality_id)

        for name, p in zip(param_names, trainable_params):
            if name in fisher and name in old_params:
                total_loss += self.lambda_base * (fisher[name] * (p - old_params[name]).pow(2)).sum()

        return total_loss


class DynamicModalityLoRALinear(nn.Module):

    def __init__(self, original_linear, rank, alpha, initial_modalities, max_modalities):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.max_modalities = max_modalities

        for p in self.original_linear.parameters():
            p.requires_grad = False

        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()

        for _ in range(initial_modalities):
            self._add_modality_lora()

        self.current_modality = 0
        self.num_modalities = initial_modalities

    def _add_modality_lora(self):
        lora_a = nn.Linear(self.original_linear.in_features, self.rank, bias=False)
        lora_b = nn.Linear(self.rank, self.original_linear.out_features, bias=False)
        nn.init.normal_(lora_a.weight, std=0.02)
        nn.init.zeros_(lora_b.weight)
        self.lora_A.append(lora_a)
        self.lora_B.append(lora_b)

    def ensure_modality_exists(self, modality_id):
        while len(self.lora_A) <= modality_id:
            self._add_modality_lora()
            self.num_modalities = len(self.lora_A)

    def set_modality(self, modality_id):
        self.ensure_modality_exists(modality_id)
        self.current_modality = modality_id

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = self.lora_B[self.current_modality](
            self.lora_A[self.current_modality](x)
        ) * self.scaling
        return base_out + lora_out

    def get_modality_params(self, modality_id):
        self.ensure_modality_exists(modality_id)
        return list(self.lora_A[modality_id].parameters()) + list(self.lora_B[modality_id].parameters())


class TaskAdapter(nn.Module):

    def __init__(self, bottleneck_dim):
        super().__init__()
        self.visual_adapter = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.text_adapter = nn.Sequential(
            nn.Linear(512, bottleneck_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, 512)
        )

    def forward(self, logits, text_features=None):
        logits_4d = logits.unsqueeze(1) if len(logits.shape) == 3 else logits
        enhanced = self.visual_adapter(logits_4d).squeeze(1)
        if text_features is not None:
            return enhanced, self.text_adapter(text_features)
        return enhanced + logits


class CLIPModalityLoRAModel(nn.Module):

    def __init__(self, clipseg_model, processor, config):
        super().__init__()
        self.clipseg_backbone = clipseg_model
        self.processor = processor
        self.config = config

        for p in self.clipseg_backbone.parameters():
            p.requires_grad = False

        self.lora_rank = config.get('lora_rank', 8)
        self.lora_alpha = config.get('lora_alpha', 16)
        self.initial_modalities = config.get('initial_modalities', 2)
        self.max_modalities = config.get('max_modalities', 10)

        self.lora_modules = []
        self.lora_module_names = []
        self._apply_dynamic_lora()

        self.device = config['device']

        self.modality_manager = AdaptiveCRPModalityManager(
            self.clipseg_backbone.clip.text_model,
            processor.tokenizer, config, self.device
        )

        self.ewc_manager = EWCManager(
            lambda_base=config.get('lambda_base', 5000.0),
            fisher_samples=config.get('fisher_samples', 200),
            config=config
        )

        self.task_adapters = nn.ModuleDict()
        self.task_seg_heads = nn.ModuleDict()
        self.current_task_id = 0
        self.current_modality_id = 0

        self.modality_global_enhancers = nn.ModuleList([
            self._create_enhancer() for _ in range(self.initial_modalities)
        ])

        self.image_size = config['image_size']
        self.to(device=self.device)

    def _create_enhancer(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def _ensure_modality_components(self, modality_id):
        while len(self.modality_global_enhancers) <= modality_id:
            new_enhancer = self._create_enhancer().to(self.device)
            self.modality_global_enhancers.append(new_enhancer)

        for lora_module in self.lora_modules:
            lora_module.ensure_modality_exists(modality_id)
            lora_module.lora_A[modality_id] = lora_module.lora_A[modality_id].to(self.device)
            lora_module.lora_B[modality_id] = lora_module.lora_B[modality_id].to(self.device)

    def _apply_dynamic_lora(self):
        vision_targets = ['q_proj', 'v_proj', 'k_proj', 'out_proj']
        text_targets = ['q_proj', 'v_proj', 'k_proj', 'out_proj']
        other_targets = ['clip_project', 'reduce']

        def apply_recursive(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear):
                    should_apply = any(t in full_name for t in other_targets)
                    if 'vision' in full_name.lower() and any(t in full_name for t in vision_targets):
                        should_apply = True
                    if 'text' in full_name.lower() and any(t in full_name for t in text_targets):
                        should_apply = True
                    if should_apply:
                        dynamic_lora = DynamicModalityLoRALinear(
                            child, self.lora_rank, self.lora_alpha,
                            self.initial_modalities, self.max_modalities
                        )
                        setattr(module, name, dynamic_lora)
                        self.lora_modules.append(dynamic_lora)
                        self.lora_module_names.append(full_name)
                else:
                    apply_recursive(child, full_name)

        apply_recursive(self.clipseg_backbone)

    def set_modality(self, modality_id):
        self._ensure_modality_components(modality_id)
        self.current_modality_id = modality_id
        for lora_module in self.lora_modules:
            lora_module.set_modality(modality_id)

    def add_task_components(self, task_id, reuse_from=None):
        key = f"task_{task_id}"
        if reuse_from is not None:
            src = f"task_{reuse_from}"
            self.task_adapters[key] = self.task_adapters[src]
            self.task_seg_heads[key] = self.task_seg_heads[src]
        else:
            self.task_adapters[key] = TaskAdapter(bottleneck_dim=64).to(self.device)
            self.task_seg_heads[key] = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 2, 1)
            ).to(self.device)

    def start_new_task(self, prompt, dataset_name, dataloader):
        modality_id, distance, is_new, representative_prompt = \
            self.modality_manager.assign_modality(self.current_task_id, dataloader)

        self.ewc_manager.register_task(self.current_task_id, modality_id)

        max_sim = 1.0 - distance
        n_modalities = self.modality_manager.get_num_modalities()
        print(f"  --> Task {self.current_task_id} -> M{modality_id} "
              f"({'NEW' if is_new else 'JOIN'}), sim={max_sim:.3f}, "
              f"{n_modalities} modalities")

        self.set_modality(modality_id)
        first_task = self.modality_manager.get_first_task_in_modality(modality_id)

        if is_new:
            self.add_task_components(self.current_task_id)
        elif first_task is not None and first_task != self.current_task_id:
            self.add_task_components(self.current_task_id, reuse_from=first_task)
        else:
            self.add_task_components(self.current_task_id)

        tid = self.current_task_id
        self.current_task_id += 1
        return tid, modality_id

    def finish_task(self, dataloader, device, task_id, modality_id):
        self.ewc_manager.consolidate(self, dataloader, device, task_id, modality_id, self.config)

    def forward(self, inputs, prompts=None, task_id=None):
        outputs = self.clipseg_backbone(**inputs)
        logits = outputs.logits
        text_feat = (outputs.text_model_output.pooler_output
                     if hasattr(outputs, 'text_model_output') and outputs.text_model_output
                     else None)

        enh_in = logits.unsqueeze(1) if len(logits.shape) == 3 else logits
        if enh_in.dim() == 2:
            enh_in = enh_in.unsqueeze(0).unsqueeze(0)
        elif enh_in.dim() == 3:
            if enh_in.shape[0] == 352:
                enh_in = enh_in.permute(2, 0, 1).unsqueeze(0)
            else:
                enh_in = enh_in.unsqueeze(1)

        enhancer = self.modality_global_enhancers[self.current_modality_id]
        enhanced = logits + enhancer(enh_in).squeeze(1)

        key = f"task_{task_id}" if task_id is not None else f"task_{self.current_task_id - 1}"

        if key in self.task_adapters:
            if text_feat is not None:
                enhanced, adapted_text = self.task_adapters[key](enhanced, text_feat)
                enhanced = enhanced + adapted_text.mean(dim=1, keepdim=True).expand_as(enhanced) * 0.1
            else:
                enhanced = self.task_adapters[key](enhanced)

        enhanced = enhanced.unsqueeze(1) if len(enhanced.shape) == 3 else enhanced
        final = self.task_seg_heads[key](enhanced) if key in self.task_seg_heads else enhanced

        if final.shape[-2:] != (self.image_size, self.image_size):
            final = F.interpolate(final, size=(self.image_size, self.image_size), mode='bilinear')

        return {'logits': final, 'text_features': text_feat}

    def get_trainable_parameters(self, modality_id=None):
        if modality_id is None:
            modality_id = self.current_modality_id
        self._ensure_modality_components(modality_id)

        params = list(self.modality_global_enhancers[modality_id].parameters())
        for lora_module in self.lora_modules:
            params.extend(lora_module.get_modality_params(modality_id))

        key = f"task_{self.current_task_id - 1}"
        if key in self.task_adapters:
            params.extend(self.task_adapters[key].parameters())
        if key in self.task_seg_heads:
            params.extend(self.task_seg_heads[key].parameters())

        return params

    def get_trainable_param_names(self, modality_id=None):
        if modality_id is None:
            modality_id = self.current_modality_id
        self._ensure_modality_components(modality_id)

        names = []
        for i, _ in enumerate(self.modality_global_enhancers[modality_id].parameters()):
            names.append(f"enhancer_M{modality_id}.{i}")

        for lora_name, lora_module in zip(self.lora_module_names, self.lora_modules):
            for i, _ in enumerate(lora_module.lora_A[modality_id].parameters()):
                names.append(f"{lora_name}.lora_A.{modality_id}.{i}")
            for i, _ in enumerate(lora_module.lora_B[modality_id].parameters()):
                names.append(f"{lora_name}.lora_B.{modality_id}.{i}")

        current_task = self.current_task_id - 1
        if f"task_{current_task}" in self.task_adapters:
            for i, _ in enumerate(self.task_adapters[f"task_{current_task}"].parameters()):
                names.append(f"adapter.task_{current_task}.{i}")
        if f"task_{current_task}" in self.task_seg_heads:
            for i, _ in enumerate(self.task_seg_heads[f"task_{current_task}"].parameters()):
                names.append(f"seg_head.task_{current_task}.{i}")

        return names

    def get_ewc_loss(self):
        return self.ewc_manager.get_ewc_loss(self, self.current_modality_id)

    def get_analysis_info(self):
        modality_info = self.modality_manager.get_analysis_info()

        lora_params_per_modality = 0
        if self.lora_modules:
            for lora in self.lora_modules:
                if len(lora.lora_A) > 0:
                    lora_params_per_modality += sum(p.numel() for p in lora.get_modality_params(0))

        enhancer_params = (sum(p.numel() for p in self.modality_global_enhancers[0].parameters())
                          if self.modality_global_enhancers else 0)

        task_params = 0
        for adapter in self.task_adapters.values():
            task_params += sum(p.numel() for p in adapter.parameters())
        for head in self.task_seg_heads.values():
            task_params += sum(p.numel() for p in head.parameters())

        num_modalities = self.modality_manager.get_num_modalities()
        total_modality_params = (lora_params_per_modality + enhancer_params) * num_modalities
        trainable = total_modality_params + task_params
        total = sum(p.numel() for p in self.parameters())

        return {
            'current_task': self.current_task_id,
            'trainable_params': trainable,
            'total_params': total,
            'lora_params_per_modality': lora_params_per_modality,
            'enhancer_params_per_modality': enhancer_params,
            'task_params': task_params,
            'num_modalities': num_modalities,
            'efficiency_ratio': trainable / total if total > 0 else 0,
            'modality_analysis': modality_info
        }