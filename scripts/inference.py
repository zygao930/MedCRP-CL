import os
import sys
import torch
import numpy as np
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from train import load_model_full, evaluate_task
from task_orders import get_all_orders

def main():
    checkpoint_path = sys.argv[1]

    config = {
        'data_root': './data',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'image_size': 352,
        'text_max_length': 77,
        'eval_batch_size': 12,
        'num_workers': 4,
    }

    model, processor = load_model_full(checkpoint_path, config)
    model.eval()

    datasets = get_all_orders()['scale_interleaved']

    all_dices = []
    for task_id, ds_info in enumerate(datasets):
        dice, loss = evaluate_task(
            model, processor, config['data_root'], ds_info['name'],
            config['device'], config, 'concise', task_id
        )
        mid = model.modality_manager.task_modality_id.get(task_id, -1)
        print(f"  {ds_info['name']} (M{mid}): {dice:.4f}")
        all_dices.append(dice)

    print(f"\nAverage Dice: {np.mean(all_dices):.4f}")


if __name__ == "__main__":
    main()