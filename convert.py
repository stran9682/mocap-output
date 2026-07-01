import pickle
import numpy as np

result = pickle.load(open('00005_prediction_result.pkl', 'rb'))
pred_output = result['pred_output_list'][0]

for key in pred_output.keys():
    print(key, np.array(pred_output[key]).shape)

betas = [pred_output['pred_betas'][0]]
global_orient = [pred_output['pred_body_pose'][0][:3]]
body_pose = [pred_output['pred_body_pose'][0][3:66]]
left_hand_pose = [pred_output['pred_left_hand_pose'][0]]
right_hand_pose = [pred_output['pred_right_hand_pose'][0]]
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

# keys ['betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'transl', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']
sample = pickle.load(open('10069_m_Kenneth_0_0.pkl', 'rb'))
test = pickle.load(open('body_pose.pkl', 'rb'))

for key in sample.keys():
    print(key, np.array(sample[key]).shape)
    print(key, np.array(test[key]).shape)