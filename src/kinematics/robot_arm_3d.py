import os
import pickle
import time
import numpy as np
from rokin.Robots import JustinArm07
# Other available robots are JustinFinger03, JustinHand12, Justin19

def pdist(a, b):
    """Pairwise distance"""
    return np.linalg.norm(a-b, axis=1)

def generate_data(robot, n=5, use_rot = False, inverses_each: int = 10, epsilon: float = 5e-2, mult:float = 1000, out_dir = "data", filename: str = "inverse_data"):
    start = time.time()
    priors = robot.sample_q(shape=n)

    # Forward
    # calculate frames of shape (NUM_SAMPLES, n_frames, 4, 4). (one homogeneous matrix for each configuration)
    frames = robot.get_frames(priors)
    # we are only interested in the last frame for each configuration, since it corresponds to the TCP
    tcp_frames = frames[:, -1, :, :]
    # the position of the TCP are the first three entries in the final column of each frame
    pos_tcp = tcp_frames[:, 0:3, 3]
    rot_tcp = tcp_frames[:, 0:3, 0:3]
    #TODO: lookup why there are no limb_lengths for JustinArm07 
    # --> how to get total length of arm to scale epsilon appropriately?

    thetas = np.zeros((n*inverses_each, robot.n_dof))

    for i in range(n):
        num_thetas_close = 0

        #TODO: decide if we want to check for the generated data if the 
        # values are without collision checks 

        while num_thetas_close < inverses_each:
            # Sample thetas, forward, keep thetas with position close to target
            thetas_gen = robot.sample_q(shape=mult*inverses_each)
            frames_gen = robot.get_frames(thetas_gen)
            # We are only interested in the tcp frames
            frames_tcp_gen = frames_gen[:, -1, :, :]
            pos_gen = frames_tcp_gen[:, 0:3, 3]
            rot_gen = frames_tcp_gen[:, 0:3, 0:3]

            thetas_close = thetas_gen[pdist(pos_tcp[i], pos_gen) <= epsilon]

            # TODO: implement metric for rotation
            num_thetas_close_batch = thetas_close.shape[0]
            if num_thetas_close_batch == 0:
                print(f"cont{i}")
                continue
            # Fill with thetas that yield a close position
            if num_thetas_close + num_thetas_close_batch > inverses_each:
                num_diff = inverses_each - num_thetas_close
                thetas[i*inverses_each + num_thetas_close : (i+1)*inverses_each] = thetas_close[0:num_diff]
                print(i)
                break
            elif num_thetas_close_batch > 0:
                thetas[i*inverses_each + num_thetas_close : i*inverses_each + num_thetas_close + num_thetas_close_batch] = thetas_close[:]
            num_thetas_close += num_thetas_close_batch

    end = time.time()

    # Appending information to filename and saving
    filename += f"_{robot.id}_{pos_tcp.shape[0]}_{inverses_each}"
    print(f"Time taken for data generation: {end-start:.3f} seconds, generated:\nRobot: {robot.id} with {pos_tcp.shape[0]} forward configurations with {inverses_each} inverse configurations each.")
    # Save data
    if use_rot:
        data = {
            "pos": pos_tcp,
            "rot": rot_tcp,
            "thetas": thetas
        }  
    else:
        data = {
            "pos": pos_tcp,
            "thetas": thetas
        }
    os.makedirs(out_dir, exist_ok=True)
    path = out_dir + os.path.sep + filename + ".pickle"
    with open(path, "wb") as file:
        pickle.dump(data, file)
    print(f"Saved the data under {path}")

if __name__ == "__main__":
    robot = JustinArm07()
    # n=1000, inverses_each=100 takes about 1344s on my VM
    generate_data(robot, n=1000, inverses_each=100)
