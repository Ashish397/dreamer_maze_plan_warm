import random
import torch
import cv2

import matplotlib.pyplot as plt
import numpy as np

from tensordict import TensorDict, pad
from pathlib import Path


def load_dataset(dataset_name, original_shape, reshap_leng, path_root):
    dataset_path = path_root + '//' + dataset_name
    dataset =  TensorDict.load_memmap(dataset_path).reshape(original_shape[0],-1)
    dataset['steps'] = torch.arange(original_shape[1]).repeat(original_shape[0],1).unsqueeze(-1)
    return dataset.reshape(int(original_shape[0]*(original_shape[1]/reshap_leng)), -1)
    
def batch_processor(data, lookahead):
    batch_size, traj_len = data.shape[:2]
    
    # Extract sequences in a batched manner
    ss = torch.stack([data["state_enc"][i] for i in range(batch_size)])
    aa = torch.stack([data["action"][i] for i in range(batch_size)]) * 0.9999 # to avoid over / underflow
   
    # Done and padding mask
    padding_mask = torch.cat(
        [
            torch.ones((batch_size, traj_len - lookahead,),),
            torch.zeros((batch_size, lookahead - 1,),),
            torch.ones((batch_size, 1,),),
        ],
        dim=1,
    )

    return ss, aa, padding_mask

def loss_fn(
    a_hat_dist,
    a,
    attention_mask,
    entropy_reg
    ):
    # a_hat is a SquashedNormal Distribution
    log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

    entropy = a_hat_dist.entropy().mean()
    loss = -(log_likelihood + entropy_reg * entropy)

    return (
        loss,
        -log_likelihood,
        entropy,
    )

def process(vid):
    try:
        vid = vid.detach().cpu().numpy()
    except AttributeError:
        None
    vid = vid - np.min(vid)
    vid = (vid / np.max(vid) * 255).astype('uint8')
    if vid.shape[-1] != 3:
        vid = np.transpose(vid, [0, 2, 3, 1])
    return vid

def make_vid(video_deck, vidname, suffix ='', num=None, fps=20):
    if num is None:
        video = cv2.VideoWriter(vidname+'//'+suffix+'.avi', 0, fps, (video_deck.shape[2],video_deck.shape[1]))
    else:
        video = cv2.VideoWriter(vidname+'//'+suffix+str(num)+'.avi', 0, fps, (video_deck.shape[2],video_deck.shape[1]))

    for image in video_deck:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()

def resize_video(video, new_size):
    """Resizes each frame in the video to the new size."""
    resized_frames = []
    for frame in video:
        resized_frame = cv2.resize(frame, [new_size[1], new_size[0]], interpolation=cv2.INTER_CUBIC)
        resized_frames.append(resized_frame)
    return np.array(resized_frames)

def combine_vids(vid_1, vid_2):
    new_size = (np.array(vid_1.shape[1:3]) / 2).astype(int)
    concat_dim = 1 + np.argmin(vid_1.shape[1:3])
    vid_1 = resize_video(vid_1, new_size)
    vid_2 = resize_video(vid_2, new_size)
    return np.concatenate([vid_1, vid_2], axis=concat_dim)

def combine_videos_in_grid(videos):
    """
    Combines a list of videos into a grid using numpy concatenate operations.

    Parameters:
    - videos: list of videos (each video is a numpy array of shape [frames, height, width, channels])
              The number of videos must be a perfect square (e.g., 4, 9, 16, etc.).

    Returns:
    - combined video in grid layout as a numpy array
    """
    num_videos = len(videos)
    grid_size = int(np.sqrt(num_videos))  # Determine grid size (e.g., for 16 videos, grid size is 4)

    # Ensure all videos have the same shape
    frames, height, width, channels = videos[0].shape

    # Reshape into rows
    rows = []
    for i in range(grid_size):
        # Concatenate videos horizontally to form a row
        row_videos = videos[i * grid_size:(i + 1) * grid_size]
        row = np.concatenate(row_videos, axis=2)  # Concatenate along width (axis=2)
        rows.append(row)

    # Concatenate rows vertically to form the final grid
    grid_video = np.concatenate(rows, axis=1)  # Concatenate along height (axis=1)

    return grid_video

def proc_nex_st(st, pae, ne, sd):
    st['pixels'] = st[('next', 'pixels')]
    st['done'] = st[('next', 'done')]
    st['terminated'] = st[('next', 'terminated')]
    st['truncated'] = st[('next', 'truncated')]
    st['step_count'] = st[('next', 'step_count')]
    st.pop('next')
    return process_and_concat_state(st, pae, ne, sd)

def process_and_concat_state(st, proc_ae, num_envs, state_dim, first_step=False):
    # Process state with proc_ae and move to the correct device
    st = proc_ae(st)#.to(device)
    # st.pop('state_dec')
    # Reshape 'state_enc' and 'pixels'
    state_enc_reshaped = st['state_enc'].reshape(num_envs, -1, state_dim)
    state_enc_reshaped = state_enc_reshaped + 4.2136
    state_enc_reshaped = torch.log(state_enc_reshaped) * 0.09
    state_enc_reshaped = state_enc_reshaped - 1
    pixels_reshaped = st['pixels'].reshape(num_envs, -1, 3, 64, 128)
    state_dec_reshaped = st['state_dec'].reshape(num_envs, -1, 3, 64, 128)
    done_reshaped = st['done'].reshape(num_envs, -1, 1)
    term_reshaped = st['terminated'].reshape(num_envs, -1, 1)
    trun_reshaped = st['truncated'].reshape(num_envs, -1, 1)
    coun_reshaped = st['step_count'].reshape(num_envs, -1, 1)

    if first_step:
        # Initialize 'state_encs' and 'pixelss' on the first step
        st['state_encs'] = state_enc_reshaped
        st['pixelss'] = pixels_reshaped
        st['state_decs'] = state_dec_reshaped
        st['dones'] = done_reshaped
        st['terms'] = term_reshaped
        st['truns'] = trun_reshaped
        st['step_counts'] = coun_reshaped
    else:
        # Concatenate to the existing 'state_encs' and 'pixelss' fields
        st['state_encs'] = torch.cat([st['state_encs'], state_enc_reshaped], dim=1)
        st['pixelss'] = torch.cat([st['pixelss'], pixels_reshaped], dim=1)
        st['state_decs'] = torch.cat([st['state_decs'], state_dec_reshaped], dim=1)
        st['dones'] = torch.cat([st['dones'], done_reshaped], dim=1)
        st['terms'] = torch.cat([st['terms'], term_reshaped], dim=1)
        st['truns'] = torch.cat([st['truns'], trun_reshaped], dim=1)
        st['step_counts'] = torch.cat([st['step_counts'], coun_reshaped], dim=1)
    
    # Clean up intermediate fields
    st.pop('state_enc')
    st.pop('pixels')
    st.pop('state_dec')
    st.pop('terminated')
    st.pop('truncated')
    st.pop('done')
    # st.pop('step_count')
    return st

def plot_rewards(all_rewards, path, window_size=20):
    """
    Plots the rewards with labeled axes and a legend, computes and plots the running mean,
    and saves the plot to the specified path.

    Args:
        all_rewards (list): A list of [non_mean_rew, mean_rew] pairs.
        path (str): The path where the plot will be saved.
        window_size (int): The size of the window for calculating the running mean.
    
    """
    # Convert the list of rewards to a numpy array for easier manipulation
    rewards_array = np.array(all_rewards)
    
    # Compute the running mean with the specified window size
    running_mean = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
    
    # Plot the original rewards
    plt.clf()
    plt.plot(rewards_array, label='Original Rewards')
    
    # Plot the running mean
    plt.plot(np.arange(window_size - 1, len(all_rewards)), running_mean, label=f'Running Mean (window size={window_size})', color='orange')
    
    # Adding labels to the axes
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    
    # Adding a legend
    plt.legend()
    
    # Save the plot with the specified path
    plt.savefig(path + '//' + 'test_rewards')
    plt.clf()

def plot_running_mean_with_std(all_outputs, star_ind=0, end_ind=-1, window_size=20, path=None, fname_prefix=None):
    """
    Plots the running mean line with a shaded area representing two standard deviations.

    Args:
        all_outputs (list of dicts): A list of dictionaries containing the output values.
        star_ind (int, optional): The starting index for slicing the data.
        end_ind (int, optional): The ending index for slicing the data.
        window_size (int, optional): The window size for calculating the running mean.
        path (str, optional): If provided, the path where the plot will be saved.
        fname_prefix (str, optional): Prefix for the filename if saving the plot.
    """
    # Extract the mean values
    mean_array = [ii['training/train_loss_mean'] for ii in all_outputs]
    
    # Convert input to numpy arrays if they aren't already
    mean_array = np.array(mean_array)
    mean_array = mean_array[star_ind:end_ind]
    
    # Calculate the running mean
    running_mean = np.convolve(mean_array, np.ones(window_size)/window_size, mode='valid')
    
    # Plotting the running mean line
    plt.plot(running_mean, label='Running Mean (Window Size: {})'.format(window_size))
    
    # Adding labels and title
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss with Running Mean")
    plt.legend()

    # Save the plot if a path is provided
    if path:
        plt.savefig(path + '//' + fname_prefix + 'train_loss_running_mean.png')
    else:
        # Show the plot
        plt.show()

def popper(tens, keys_to_keep):
    tens_keys = list(tens.keys())
    for key in tens_keys:
        if key not in keys_to_keep:
            tens.pop(key)
    return tens

def slice_trajectories(this_data, K):
    """
    Slices trajectories from the given TensorDict `this_data` based on the configuration in `variant`.

    Args:
        this_data (TensorDict): The input TensorDict containing trajectories.
        K (int): determines the length of the slice

    Returns:
        TensorDict: A new TensorDict containing the sliced trajectories.
    """

    num_traj, _ = this_data.batch_size

    # Calculate random start indices
    start_indices = (torch.rand(num_traj) * (torch.where(this_data['done'])[1] + 1 - K).squeeze()).int()

    # Create a list to store the sliced TensorDicts
    sliced_trajectories = []

    for traj_num in range(num_traj):
        # Slice the TensorDict and add it to the list
        sliced_trajectories.append(this_data[traj_num][start_indices[traj_num]:start_indices[traj_num] + K])

    # Stack the sliced TensorDicts into a new TensorDict
    sliced_data = torch.stack(sliced_trajectories)

    return sliced_data


def slice_for_test(this_data, size, max_len):
    outp = this_data[:,:size]
    if size > max_len:
        outp = outp[:,-max_len:]
    elif size < max_len:
        outp = pad(outp, [0,0,int(max_len-size),0], value=0)
    return outp

def noise_maker(shape, num_centers, ks, x_radius, y_radius, mean, std, overlap=0, max_retries=100):
    # Create a zero tensor for the full frame (shape is batch_size, timesteps, state_dim)
    full_frame = torch.zeros(shape)

    # Precompute the grid of distances for the given radius
    y_range = torch.arange(-y_radius, y_radius + 1)
    x_range = torch.arange(-x_radius, x_radius + 1)
    x_dist, y_dist = torch.meshgrid(x_range, y_range, indexing='ij')
    distance = torch.max(torch.abs(x_dist), torch.abs(y_dist))
    weights = 1 / (0.001 * distance + 1)#used to be 0.1

    # Iterate over each batch
    for i in range(shape[0]):
        # Track the x coordinates of centers (second-to-last dimension) if overlap is not allowed
        used_x_centers = [] if overlap == 0 else None

        # Generate random center coordinates for each trajectory
        centers = []
        for _ in range(num_centers):
            retries = 0
            valid_center_found = False
            
            while not valid_center_found and retries < max_retries:
                center_x = random.randint(ks[i], shape[-2] - 1)  # Time dimension center
                
                # If overlap is not allowed, ensure the centers are spaced at least `2 * x_radius` apart
                if overlap == 1 or all(abs(center_x - used_x) > 2 * x_radius for used_x in used_x_centers):
                    center_y = random.randint(0, shape[-1] - 1)  # State dimension center
                    centers.append((center_x, center_y))
                    if overlap == 0:
                        used_x_centers.append(center_x)
                    valid_center_found = True
                
                retries += 1
            
            # Break if no valid center was found after max retries
            if retries == max_retries:
                # print(f"Warning: Could not place all centers in batch {i}, reduced to {len(centers)} centers.")
                break  # Stop adding centers if it's too difficult to place more

        # For each center, apply weighted mask using broadcasting
        for center_x, center_y in centers:
            # Define valid x and y coordinates based on the center and radius
            x_min = max(ks[i], center_x - x_radius)
            x_max = min(shape[-2], center_x + x_radius + 1)
            y_min = max(0, center_y - y_radius)
            y_max = min(shape[-1], center_y + y_radius + 1)
            
            # Get slice of weights that fit within the valid range
            weight_slice = weights[
                x_radius - (center_x - x_min):x_radius + (x_max - center_x),
                y_radius - (center_y - y_min):y_radius + (y_max - center_y)
            ]
            
            # Add weights to the full frame tensor
            full_frame[i, x_min:x_max, y_min:y_max] += weight_slice

    # Apply noise to the full frame
    full_frame = full_frame * torch.normal(mean=mean, std=std, size=shape)
    
    # Replace zeros with ones
    full_frame[full_frame == 0] = 1

    return full_frame


def plot_running_mean(values, suffix=None, path = None, x_label='epochs', window_size=20):
    """
    Plots the given list of values with labeled axes, computes and plots the running mean,
    and saves the plot to the specified path with the given suffix.

    Args:
        values (list or array-like): A list of numerical values to be plotted.
        path (str): The path where the plot will be saved.
        suffix (str): A string suffix to name the saved file.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        label (str): The label for the original values in the plot.
        window_size (int): The size of the window for calculating the running mean.
    """
    # Convert the list of values to a numpy array for easier manipulation
    values_array = np.array(values)
    
    # Compute the running mean with the specified window size
    running_mean = np.convolve(values_array, np.ones(window_size) / window_size, mode='valid')
    
    # Plot the original values
    plt.clf()  # Clear the current figure
    plt.plot(values_array, label=suffix)
    
    # Plot the running mean
    plt.plot(np.arange(window_size - 1, len(values)), running_mean, 
             label=f'Running Mean (window size={window_size})', color='orange')
    
    # Adding labels to the axes
    plt.xlabel(x_label)
    if suffix is not None:
        plt.ylabel(suffix)
    
    # Adding a legend
    plt.legend()
    
    if path:
        # Save the plot to the specified path with the suffix in the filename
        plt.savefig(f"{path}//{suffix}_ws_{window_size}_plot.png")
    else:
        plt.show()
    plt.clf()  # Clear the figure after saving

def plot_sum_stats(sum_stats, path, window_size=20):
    tlm = [to['training/train_loss_mean'] for to in sum_stats]
    tls = [to['training/train_loss_std'] for to in sum_stats]
    nll = [to['training/nll'][0] for to in sum_stats]
    ent = [to['training/entropy'][0] for to in sum_stats]
    tv = [to['training/temp_value'] for to in sum_stats]

    plot_running_mean(values=tlm, path=path, suffix='training_loss_mean', x_label='timesteps', window_size=window_size)
    plot_running_mean(values=tls, path=path, suffix='train_loss_std', x_label='timesteps', window_size=window_size)
    plot_running_mean(values=tv, path=path, suffix='temp_value', x_label='timesteps', window_size=window_size)
    plot_running_mean(values=ent, path=path, suffix='entropy', x_label='timesteps', window_size=window_size)
    plot_running_mean(values=nll, path=path, suffix='nll', x_label='timesteps', window_size=window_size)


def evaluate_fn(
    model,
    outs,
    proc_ae,
    use_mean=0,
    ):

    model.eval()

    (
        states,
        actions,
        padding_mask,
    ) = outs

    with torch.no_grad():
        states = states.to('cuda')
        state_target = torch.clone(states)

        states[...,-1,:1024] = 0

        actions = actions.to('cuda')
        padding_mask = padding_mask.to('cuda')

        action_target = torch.clone(actions)

        state_preds, action_preds, _ = model.forward(
            states,
            actions,
            None,
            None,
            None,
            None,
            padding_mask,
        )

        loss_a, nll_a, entropy_a = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            model.temperature().detach(),  # no gradient taken here
        )
        loss_s, nll_s, entropy_s = loss_fn(
            state_preds,  # s_hat_dist
            state_target,
            padding_mask,
            model.temperature().detach(),  # no gradient taken here
        )
        loss = loss_s + loss_a * 0.1
        nll = nll_s + nll_a * 0.1
        entropy = entropy_s + entropy_a * 0.1

        loss = loss.cpu().item(),
        nll = nll.cpu().item(),
        entropy = entropy.cpu().item(),

        states = states.cpu()
        actions = actions.cpu()
        padding_mask = padding_mask.cpu()
        action_target = action_target.cpu()
        state_target = action_target.cpu()

        if use_mean:
            preds = state_preds.mean
        else:
            preds = state_preds.sample()
        preds = (torch.exp((preds + 1) / 0.09)) - 4.2136
        with proc_ae.ae_net_params.to_module(proc_ae.ae_net):
            td_copy = proc_ae.ae_net.recreate(preds[0])
        td_copy = np.transpose(td_copy.detach().cpu().numpy(), [0,2,3,1])

        del(action_preds)
        del(states)
        del(actions)
        del(padding_mask)
        del(action_target)
        del(state_target)

        return (
            loss,
            nll,
            entropy,
            td_copy
        )