import cnn_ngram
import mask_out
import os, sys
from shutil import copyfile


def iterate(init_round, mask_val_flag, max_iter):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(init_round, max_iter):
        print("===================Round %d===================" % i)
        # Train the CNN
        print("===================Train model===================")
        cnn_ngram.train()
        print("===================Save model===================")
        # Copy the model to the designated directory
        save_model_path = os.path.join(cur_dir, 'new_models/round%d' % i)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        copyfile('weights.best.hdf5', os.path.join(save_model_path, 'weights.best.hdf5'))
        # Mask out sentences
        print("===================Mask sentences===================")
        mask_out.run(str(i), mask_val_flag, verbose=0)


if __name__ == '__main__':
    start_round = int(sys.argv[1])
    val_flag = sys.argv[2]
    max_iteration = int(sys.argv[3])
    if val_flag == "true":
        val_flag = True
    else:
        val_flag = False
    iterate(start_round, val_flag, max_iteration)
