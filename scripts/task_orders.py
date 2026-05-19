MEDICAL_TASK_ORDERS = {
    # 'scale_grouped': [
    #     # {'name': 'camus_lv', 'description': 'Cardiac ultrasound segmentation'},
    #     # {'name': 'camus_la', 'description': 'Cardiac ultrasound segmentation'},
    #     # {'name': 'camus_myo', 'description': 'Cardiac ultrasound segmentation'},
    #     # Cardiac US (1 combined)
    #     {'name': 'camus', 'description': 'Cardiac ultrasound segmentation'},
    #     # Endoscopy (5)
    #     {'name': 'kvasir_polyp', 'description': 'Endoscopic polyp segmentation'},
    #     {'name': 'clinicdb_polyp', 'description': 'Endoscopic polyp segmentation'},
    #     {'name': 'etis_polyp', 'description': 'Endoscopic polyp segmentation'},
    #     {'name': 'cvc300_polyp', 'description': 'Endoscopic polyp segmentation'},
    #     {'name': 'colondb_polyp', 'description': 'Endoscopic polyp segmentation'},
    #     # Skin (1)
    #     {'name': 'isic', 'description': 'Skin lesion segmentation'},
    #     # X-Ray (7)
    #     {'name': 'chex_airspace_opacity', 'description': 'Chest X-ray airspace opacity'},
    #     {'name': 'chex_atelectasis', 'description': 'Chest X-ray atelectasis'},
    #     {'name': 'chex_cardiomegaly', 'description': 'Chest X-ray cardiomegaly'},
    #     {'name': 'chex_edema', 'description': 'Chest X-ray edema'},
    #     {'name': 'chex_enlarged_cardiomediastinum', 'description': 'Chest X-ray enlarged cardiomediastinum'},
    #     {'name': 'chex_pleural_effusion', 'description': 'Chest X-ray pleural effusion'},
    #     {'name': 'chex_support_devices', 'description': 'Chest X-ray support devices'},
    #     # Breast US (2)
    #     {'name': 'busi_benign', 'description': 'Breast ultrasound - benign tumor'},
    #     {'name': 'busi_malignant', 'description': 'Breast ultrasound - malignant tumor'},
    # ],

    # 'scale_reversed': [
    #     # Breast US (2)
    #     {'name': 'busi_malignant', 'description': 'Breast ultrasound - malignant tumor'},
    #     {'name': 'busi_benign', 'description': 'Breast ultrasound - benign tumor'},
    #     # X-Ray (7)
    #     {'name': 'chex_support_devices', 'description': 'Chest X-ray support devices'},
    #     {'name': 'chex_pleural_effusion', 'description': 'Chest X-ray pleural effusion'},
    #     {'name': 'chex_enlarged_cardiomediastinum', 'description': 'Chest X-ray enlarged cardiomediastinum'},
    #     {'name': 'chex_edema', 'description': 'Chest X-ray edema'},
    #     {'name': 'chex_cardiomegaly', 'description': 'Chest X-ray cardiomegaly'},
    #     {'name': 'chex_atelectasis', 'description': 'Chest X-ray atelectasis'},
    #     {'name': 'chex_airspace_opacity', 'description': 'Chest X-ray airspace opacity'},
    #     # Skin (1)
    #     {'name': 'isic', 'description': 'Skin lesion segmentation'},
    #     # Endoscopy (5)
    #     {'name': 'colondb_polyp', 'description': 'Endoscopic polyp segmentation'},
    #     {'name': 'cvc300_polyp', 'description': 'Endoscopic polyp segmentation'},
    #     {'name': 'etis_polyp', 'description': 'Endoscopic polyp segmentation'},
    #     {'name': 'clinicdb_polyp', 'description': 'Endoscopic polyp segmentation'},
    #     {'name': 'kvasir_polyp', 'description': 'Endoscopic polyp segmentation'},
    #     # Cardiac US (1 combined)
    #     {'name': 'camus', 'description': 'Cardiac ultrasound segmentation'},
    # ],

    'scale_interleaved': [
        {'name': 'camus', 'description': 'Cardiac ultrasound segmentation'},
        {'name': 'kvasir_polyp', 'description': 'Endoscopic polyp segmentation'},
        {'name': 'isic', 'description': 'Skin lesion segmentation'},
        {'name': 'chex_airspace_opacity', 'description': 'Chest X-ray airspace opacity'},
        {'name': 'busi_benign', 'description': 'Breast ultrasound - benign tumor'},
        {'name': 'clinicdb_polyp', 'description': 'Endoscopic polyp segmentation'},
        {'name': 'chex_atelectasis', 'description': 'Chest X-ray atelectasis'},
        {'name': 'etis_polyp', 'description': 'Endoscopic polyp segmentation'},
        {'name': 'chex_cardiomegaly', 'description': 'Chest X-ray cardiomegaly'},
        {'name': 'cvc300_polyp', 'description': 'Endoscopic polyp segmentation'},
        {'name': 'busi_malignant', 'description': 'Breast ultrasound - malignant tumor'},
        {'name': 'chex_edema', 'description': 'Chest X-ray edema'},
        {'name': 'colondb_polyp', 'description': 'Endoscopic polyp segmentation'},
        {'name': 'chex_enlarged_cardiomediastinum', 'description': 'Chest X-ray enlarged cardiomediastinum'},
        {'name': 'chex_pleural_effusion', 'description': 'Chest X-ray pleural effusion'},
        {'name': 'chex_support_devices', 'description': 'Chest X-ray support devices'},
    ],

    'scale_mixed': [
        {'name': 'chex_airspace_opacity', 'description': 'Chest X-ray airspace opacity'},
        {'name': 'kvasir_polyp', 'description': 'Endoscopic polyp segmentation'},
        {'name': 'camus', 'description': 'Cardiac ultrasound segmentation'},
        {'name': 'busi_benign', 'description': 'Breast ultrasound - benign tumor'},
        {'name': 'clinicdb_polyp', 'description': 'Endoscopic polyp segmentation'},
        {'name': 'chex_cardiomegaly', 'description': 'Chest X-ray cardiomegaly'},
        {'name': 'isic', 'description': 'Skin lesion segmentation'},
        {'name': 'chex_atelectasis', 'description': 'Chest X-ray atelectasis'},
        {'name': 'etis_polyp', 'description': 'Endoscopic polyp segmentation'},
        {'name': 'busi_malignant', 'description': 'Breast ultrasound - malignant tumor'},
        {'name': 'chex_edema', 'description': 'Chest X-ray edema'},
        {'name': 'cvc300_polyp', 'description': 'Endoscopic polyp segmentation'},
        {'name': 'colondb_polyp', 'description': 'Endoscopic polyp segmentation'},
        {'name': 'chex_pleural_effusion', 'description': 'Chest X-ray pleural effusion'},
        {'name': 'chex_enlarged_cardiomediastinum', 'description': 'Chest X-ray enlarged cardiomediastinum'},
        {'name': 'chex_support_devices', 'description': 'Chest X-ray support devices'},
    ]

}


def get_all_orders():
    return MEDICAL_TASK_ORDERS
