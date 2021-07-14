import pickle

from rokin.Robots import JustinArm07


NUM_SAMPLES = 2000000
OUT_FILE_NAME = "data/test_3d.pickle"


def generate_forward():
    robot = JustinArm07()

    # generate random configurations of shape (NUM_SAMPLES, n_dof)
    priors = robot.sample_q(shape=NUM_SAMPLES)
    # calculate frames of shape (NUM_SAMPLES, n_frames, 4, 4). (one homogeneous matrix for each configuration)
    frames = robot.get_frames(priors)
    # we are only interested in the last frame for each configuration, since it corresponds to the TCP
    tcp_frames = frames[:, -1, :, :]
    # the position of the TCP are the first three entries in the final column of each frame
    tcp_positions = tcp_frames[:, 0:3, 3]

    data = {
        "priors": priors,
        "positions": tcp_positions,
    }
    with open(OUT_FILE_NAME, "wb") as out_file:
        pickle.dump(data, out_file)

    print(f"Saved priors of shape {priors.shape} and positions of shape {tcp_positions.shape}")
