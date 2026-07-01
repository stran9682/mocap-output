import pickle
import numpy as np
from pathlib import Path

def extract_data(filename, output_filename):
    result = pickle.load(open(filename, 'rb'))
    pred_output = result['pred_output_list'][0]

    extracted_data = {
        'betas': [pred_output['pred_betas'][0]],
        'global_orient': [pred_output['pred_body_pose'][0][:3]],
        'body_pose': [pred_output['pred_body_pose'][0][3:66]],
        'left_hand_pose': [pred_output['pred_left_hand_pose'][0]],
        'right_hand_pose': [pred_output['pred_right_hand_pose'][0]],
        'transl': [[0] * 3],
        'jaw_pose': [[0] * 3],
        'leye_pose': [[0] * 3],
        'reye_pose': [[0] * 3],
        'expression': [[0] * 10]
    }

    with open(f"output/{output_filename}", "wb") as file:
        pickle.dump(extracted_data, file)

dir_path = Path("input")
Path("output").mkdir(parents=True, exist_ok=True)
count = 0

arr = []
for entry in dir_path.iterdir():
    if entry.is_file() and entry.suffix == '.pkl':
        output_filename = f"body_pose_{count:05d}.pkl"
        extract_data(entry, output_filename)
        count += 1

