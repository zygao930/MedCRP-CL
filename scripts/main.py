import os
import json
import torch
import numpy as np
from pathlib import Path
from model import CLIPModalityLoRAModel
from train import set_seed, train_model, serialize_config
from task_orders import get_all_orders
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    set_seed(42)
    print("Random seed set to 42")

    config = {
        'data_root': './data',
        'output_dir': './outputs',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'clipseg_path': 'CIDAS/clipseg-rd64-refined',
        'prompt_strategy': 'concise',

        'crp_alpha': 5.0,
        'initial_modalities': 2,
        'max_modalities': 10,

        'lambda_base': 5000,
        'fisher_samples': 200,

        'lora_rank': 8,
        'lora_alpha': 16,

        'image_size': 352,
        'text_max_length': 77,
        'batch_size': 16,
        'eval_batch_size': 12,
        'num_workers': 4,

        'learning_rate': 1e-3,
        'weight_decay': 8e-5,
        'max_epochs_per_task': 60,
        'min_epochs': 15,
        'patience': 8,
        'min_delta': 0.01,
        'gradient_clip_norm': 1.0,
        'print_interval': 5,

        'loss_weights': {
            'segmentation': 1.0,
            'dice': 1.0,
            'ewc': 1.0
        },
        'save_checkpoints': False
    }

    order_configs = get_all_orders()
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    processor = CLIPSegProcessor.from_pretrained(config['clipseg_path'])

    for order_name, datasets in list(order_configs.items())[:1]:

        backbone = CLIPSegForImageSegmentation.from_pretrained(config['clipseg_path'])
        model = CLIPModalityLoRAModel(backbone, processor, config)

        perf_matrix, task_results, forgetting, analysis = train_model(
            model, processor, config, datasets, order_name
        )

        with open(Path(config['output_dir']) / f'{order_name}_results.json', 'w') as f:
            json.dump({
                'performance_matrix': perf_matrix,
                'task_results': task_results,
                'forgetting_rate': forgetting,
                'order': order_name,
                'method': 'Adaptive-CRP + LoRA + EWC',
                'crp_alpha': config['crp_alpha'],
                'config': serialize_config(config)
            }, f, indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()