import matplotlib as mpl
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import os


def viz_robot_line(pos: np.ndarray, target: np.ndarray = None, epsilon=0.05, save: bool = True, show: bool = False, viz_dir: str = "visualizations", fig_name: str = "fig_line_3d", viz_format: tuple = (".png", ".svg")):
    """Visualization of robot arm as lines between joints in 3d

    :param pos: Frame positions (n, n_dof, 3).
    :param target: Target TCP position (1, 3).
    :param epsilon: Distance to target that is desired.
    :param save: Bool, True if plot should be saved
    :param show: Bool, True if plot should be displayed
    :param viz_dir: Directory in which visualizations will be saved
    :param fig_name: Name of the figure without ending or directory, e.g. "fig1"
    :param viz_format: Formats in which the plot should be saved, e.g. (".png", ".svg") or ("png",)
    """

    if not os.path.isdir(viz_dir):
        os.makedirs(viz_dir, exist_ok=True)
    
    mpl.rcParams["legend.fontsize"] = 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if target is not None:
        ax.scatter(target[:, 0], target[:, 1], target[:, 2], c="b")
    c = "k"
    # Plot all arms, green if close to target, red otherwise, black if no target given
    for i in range(pos.shape[0]):
        if target is not None:
            dist = np.linalg.norm(target - pos[i, -1,:], axis=1)
            if dist < epsilon:
                c = "g"
            elif dist < 2*epsilon:
                c = "orange"
            else:
                c = "r"
        ax.plot(pos[i, :, 0],pos[i, :, 1],pos[i, :, 2], c=c)
        ax.scatter(pos[i, :, 0],pos[i, :, 1],pos[i, :, 2], c=c)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    title = f"Robot arms"
    if target is not None:
        dist_avg = np.sum(np.linalg.norm(target - pos[:, -1,:], axis=1))/pos.shape[0]
        title += f", d_avg = {dist_avg}"        
    ax.set_title(title)
    ax.set_xlim(*(-0.7,0.7))
    ax.set_ylim(*(-0.7,0.7))
    ax.set_zlim(*(-0.7,0.7))
    ax.legend()

    # Save
    if save:
        for format in viz_format:
            fig.savefig(os.path.join(viz_dir, fig_name) + format)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    from rokin.Robots import JustinArm07
    robot = JustinArm07()
    q = robot.sample_q(shape=100)  # Generates random configurations [shape x n_dof]
    #print(q)
    f = robot.get_frames(q)  # Calculates homogeneous matrices [n_frames x 4 x 4]
    pos = f[:,:,0:3,3]
    target = np.array([[0.4,0.4,0.4]])

    #print(pos[0].shape)
    viz_robot_line(pos, target, 0.1)
