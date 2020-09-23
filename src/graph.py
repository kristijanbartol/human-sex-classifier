import matplotlib.pyplot as plt
import numpy as np
import os


CKPT_DIR = 'checkpoint/'


def plot_graph(exp_dir):
    log_fpath = os.path.join(exp_dir, 'log.txt')
    with open(log_fpath) as flog:
        # Read all except the first line (attribute names).
        # Read all except the first two columns.
        data = np.array([[y for y in x[:-1].split()[2:]]
            for x in flog.readlines()[1:]], dtype=np.float32)
    data = np.swapaxes(data, 0, 1)
    data[[1, 3], :] *= 3.

    epochs = np.arange(0, 200, 1)

    plt.plot(epochs, data[0], label='train')
    plt.plot(epochs, data[2], label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'errors.png'))
    plt.show()
    plt.clf()

    plt.plot(epochs, data[1], label='train')
    plt.plot(epochs, data[3], label='test')
    plt.xlabel('Epochs')
    plt.ylabel('MPJPE')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'mpjpes.png'))
    plt.show()
    plt.clf()


if __name__ == '__main__':
    plot_graph(os.path.join(CKPT_DIR, 'h36m-gt'))
    plot_graph(os.path.join(CKPT_DIR, 'h36m-openpose'))

