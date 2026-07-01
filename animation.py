import pickle
import numpy as np
from pathlib import Path

def extract_frame_data(filename):
    """Extract SMPL-X parameters from a single prediction file"""
    result = pickle.load(open(filename, 'rb'))
    pred_output = result['pred_output_list'][0]
    
    return {
        'betas': pred_output['pred_betas'][0],  # (10,)
        'global_orient': pred_output['pred_body_pose'][0][:3],  # (3,)
        'body_pose': pred_output['pred_body_pose'][0][3:66],  # (63,)
        'left_hand_pose': pred_output['pred_left_hand_pose'][0],  # (45,)
        'right_hand_pose': pred_output['pred_right_hand_pose'][0],  # (45,)
    }

def create_amass_animation(input_dir='input', output_file='animation.npz', framerate=12):
    """
    Create AMASS SMPL-X format animation from sequential prediction files
    
    Args:
        input_dir: Directory containing sequential prediction pkl files
        output_file: Output NPZ filename
        framerate: FPS of the animation (default 12)
    """
    input_path = Path(input_dir)
    
    # Collect all pkl files and sort them
    pkl_files = sorted([f for f in input_path.iterdir() if f.suffix == '.pkl'])
    
    if not pkl_files:
        print(f"No pkl files found in {input_dir}")
        return
    
    print(f"Found {len(pkl_files)} frames")
    
    # Extract data from all frames
    frames_data = []
    
    for i, pkl_file in enumerate(pkl_files):
        print(f"Processing frame {i+1}/{len(pkl_files)}: {pkl_file.name}")
        frame = extract_frame_data(pkl_file)
        frames_data.append(frame)
    
    # Concatenate poses: [global_orient(3) + body_pose(63) + left_hand(45) + right_hand(45)] = 156
    poses = np.array([
        np.concatenate([
            frame['global_orient'],
            frame['body_pose'],
            frame['left_hand_pose'],
            frame['right_hand_pose']
        ])
        for frame in frames_data
    ])  # Shape: (T, 156)
    
    # Use betas from first frame (constant across sequence)
    betas = frames_data[0]['betas']  # Shape: (10,)
    
    # Translation: all zeros (stationary model)
    trans = np.zeros((len(frames_data), 3))  # Shape: (T, 3)
    
    print(f"\nAnimation summary:")
    print(f"  - Frames: {len(frames_data)}")
    print(f"  - Poses shape: {poses.shape}")
    print(f"  - Betas shape: {betas.shape}")
    print(f"  - Translation shape: {trans.shape}")
    print(f"  - Frame rate: {framerate} fps")
    print(f"  - Duration: {len(frames_data) / framerate:.2f} seconds")
    
    # Save as AMASS NPZ format
    np.savez(
        output_file,
        poses=poses,
        betas=betas,
        trans=trans,
        mocap_framerate=framerate,
        gender='neutral'
    )
    
    print(f"\n✓ Saved AMASS animation to {output_file}")

# Create animation from input directory
Path("input").mkdir(parents=True, exist_ok=True)  # Create input dir if needed
create_amass_animation(input_dir='input', output_file='animation.npz', framerate=30)