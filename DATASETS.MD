# Dataset Preparation

## Dataset Sources

| Dataset | Modality | Tasks | Source |
|---------|----------|-------|--------|
| CAMUS | Cardiac Ultrasound | 1 (multi-class: LV, LA, MYO) | http://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8 |
| Kvasir-SEG | Endoscopy | 1 | https://datasets.simula.no/kvasir-seg/ |
| ClinicDB | Endoscopy | 1 | https://www.kaggle.com/datasets/balraj98/cvcclinicdb |
| ETIS-LaribPolypDB | Endoscopy | 1 | https://drive.google.com/drive/folders/10QXjxBJqCf7PAXqbDvoceWmZ-qF07tFi?usp=sharing |
| CVC-300 | Endoscopy | 1 | https://drive.google.com/drive/folders/10QXjxBJqCf7PAXqbDvoceWmZ-qF07tFi?usp=sharing |
| CVC-ColonDB | Endoscopy | 1 | https://www.kaggle.com/datasets/hopmai/cvc-colondb |
| ISIC | Dermoscopy | 1 | https://challenge.isic-archive.com/data/ |
| CheXlocalize | Chest X-ray | 7 (see below) | https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c |
| BUSI | Breast Ultrasound | 2 (see below) | https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset |

## Multi-class Dataset Splitting

Some source datasets contain multiple classes or subtypes. We split them into separate binary segmentation tasks to enable fine-grained continual learning evaluation.

**CheXlocalize** is split into 7 pathology-specific tasks:
`chex_airspace_opacity`, `chex_atelectasis`, `chex_cardiomegaly`, `chex_edema`, `chex_enlarged_cardiomediastinum`, `chex_pleural_effusion`, `chex_support_devices`

**BUSI** is split into 2 tasks by tumor type:
`busi_benign`, `busi_malignant`

**CAMUS** is kept as a single multi-class task (`camus`) combining LV, LA, and MYO structures.

## Directory Structure

Place all datasets under `data/` at the project root. Each dataset folder follows the same structure:

```
data/
├── camus/
│   ├── anns/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   ├── images/
│   └── masks/
├── kvasir_polyp/
│   ├── anns/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   ├── images/
│   └── masks/
├── clinicdb_polyp/
├── etis_polyp/
├── cvc300_polyp/
├── colondb_polyp/
├── isic/
├── chex_airspace_opacity/
├── chex_atelectasis/
├── chex_cardiomegaly/
├── chex_edema/
├── chex_enlarged_cardiomediastinum/
├── chex_pleural_effusion/
├── chex_support_devices/
├── busi_benign/
└── busi_malignant/
```

Each folder contains:
- `anns/`: JSON annotation files with text prompts (`train.json`, `val.json`, `test.json`)
- `images/`: input images
- `masks/`: binary segmentation masks (0 = background, 255 = foreground)

The mask filenames must match the `mask_name` field in the corresponding JSON files.

## Annotation Format

Each JSON file is a list of entries:

```json
[
  {
    "img_name": "image_001.png",
    "mask_name": "mask_001.png",
    "prompts": {
      "p1": "polyp",
      ...
    }
  }
]
```

The `prompts` field is a dictionary of prompt variants at different granularity levels. The prompt strategy (`concise` or `detailed`) determines which level is used at runtime.

## Prompt Templates

Each sample has a `prompts` dictionary with multiple granularity levels. We use two strategies: `concise` (P1) and `detailed` (highest available level with fallback). The table below shows examples for all 16 tasks, organized by discovered semantic modality.

**Semantic Modality 0: Cardiac Ultrasound**

| Dataset | Concise | Detailed |
|---------|---------|----------|
| CAMUS | Left ventricular cavity in the cardiac ultrasound. | Left ventricular cavity of oval shape in four-chamber view in the cardiac ultrasound at end of the diastole cycle of a 18-year-old male with good image quality. |

**Semantic Modality 1: Endoscopy (Polyp Segmentation)**

| Dataset | Concise | Detailed |
|---------|---------|----------|
| Kvasir-SEG | round polyp | One medium pink round polyp which is a projecting growth of tissue located in top left of the image. |
| ClinicDB | oval polyp | Two large white oval polyp which is a projecting growth of tissue located in center, bottom right of the image. |
| CVC-300 | triangular polyp | One small brown triangular polyp which is a projecting growth of tissue located in bottom right of the image. |
| ColonDB | kidney polyp | One large orange kidney polyp which is a projecting growth of tissue located in center of the image. |
| ETIS | circle polyp | Two small yellow circle polyp which is a projecting growth of tissue located in top, right of the image. |

**Semantic Modality 2: Dermoscopy (Skin Lesion Segmentation)**

| Dataset | Concise | Detailed |
|---------|---------|----------|
| ISIC | circle skin melanoma | One large blue circle skin melanoma which is a dark sore with irregular texture located in center of the image. |

**Semantic Modality 3: Chest X-ray**

| Dataset | Concise | Detailed |
|---------|---------|----------|
| CheX-Airspace Opacity | Airspace Opacity in a chest Xray. | Airspace Opacity in a Chest Xray. Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity are present. |
| CheX-Atelectasis | Atelectasis in a chest Xray. | Atelectasis in a Chest Xray. Enlarged Cardiomediastinum, Lung Opacity, Atelectasis, Pleural Effusion are present. |
| CheX-Cardiomegaly | Cardiomegaly in a chest Xray. | Cardiomegaly in a Chest Xray. Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Edema are present. |
| CheX-Edema | Edema in a chest Xray. | Edema in a Chest Xray. Lung Opacity, Edema, Consolidation, Atelectasis are present. |
| CheX-Enlarged Card. | Enlarged Cardiomediastinum in a chest Xray. | Enlarged Cardiomediastinum in a Chest Xray. Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity are present. |
| CheX-Pleural Effusion | Pleural Effusion in a chest Xray. | Pleural Effusion in a Chest Xray. Lung Opacity, Atelectasis, Pleural Effusion, Support Devices are present. |
| CheX-Support Devices | Support Devices in a chest Xray. | Support Devices in a Chest Xray. Enlarged Cardiomediastinum, Cardiomegaly, Support Devices are present. |

**Semantic Modality 4: Breast Ultrasound**

| Dataset | Concise | Detailed |
|---------|---------|----------|
| BUSI-Benign | Benign tumor in the breast ultrasound image. | Five small, medium circle-shaped benign tumors at the top right, top, center, right, right in the breast ultrasound image. |
| BUSI-Malignant | Malignant tumor in the breast ultrasound image. | One large irregular-shaped malignant tumor at the center in the breast ultrasound image. |

## Note

Since these datasets are from the medical domain, you may need to comply with certain licensing conditions before downloading and using them. Please refer to the original source links for terms of use.