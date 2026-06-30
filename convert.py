import pickle
import numpy as np

result = pickle.load(open('00005_prediction_result.pkl', 'rb'))

for key in result['pred_output_list'][0].keys():
    print(key, np.array(result['pred_output_list'][0][key]).shape)

betas = [result['pred_output_list'][0]['pred_betas'][0]]
global_orient = [result['pred_output_list'][0]['pred_body_pose'][0][:3]]
body_pose = [result['pred_output_list'][0]['pred_body_pose'][0][4:67]]
left_hand_pose = [result['pred_output_list'][0]['pred_left_hand_pose'][0]]
right_hand_pose = [result['pred_output_list'][0]['pred_right_hand_pose'][0]]
transl = [[0] * 3]
jaw_pose = [[0] * 3]
leye_pose = [[0] * 3]
reye_pose = [[0] * 3]
expression = [[0] * 10]

extracted_data = {
    'betas': betas,
    'global_orient': global_orient,
    'body_pose': body_pose,
    'left_hand_pose': left_hand_pose,
    'right_hand_pose': right_hand_pose,
    'transl': transl,
    'jaw_pose': jaw_pose,
    'leye_pose': leye_pose,
    'reye_pose': reye_pose,
    'expression': expression
}

with open("body_pose.pkl", "wb") as file:
    pickle.dump(extracted_data, file)

print("="*50)

# this is correct data here
# keys ['betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'transl', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']
sample = pickle.load(open('10069_m_Kenneth_0_0.pkl', 'rb'))

for key in sample.keys():
    print(key, np.array(sample[key]).shape)