import sys, pickle, torch, glob, os
import numpy as np
from numpy.linalg import norm, svd


def count_scenes(distance_matrix):
    """
      Function to determine an estimate number of scenes in the movie given the pairwise distance matrix
    """

    singular_values = svd(distance_matrix, full_matrices=False, compute_uv=False)
    singular_values = singular_values[:len(singular_values) // 2]
    singular_values = np.log(singular_values)

    start = np.array([0, singular_values[0]])
    end = np.array([len(singular_values), singular_values[-1]])

    max_distance = 0
    count = 0

    for i, value in enumerate(singular_values):
        current = np.array([i, value])
        distance = norm(np.cross(start - end, start - current)) / norm(end - start)

        if distance > max_distance:
            max_distance = distance
            count = i

    return count


if __name__ == "__main__":

    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    filenames = ['data/' + i for i in os.listdir(data_dir)]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print('Begin optimization')
    for index, file in enumerate(filenames):
        with open(file, 'rb') as f:
            # Load each file
            data = pickle.load(f)
            print(f'File {file} loaded')
            # Features given
            features = ['place', 'cast', 'action', 'audio']
            distances = [torch.cdist(data[i], data[i], p=2) for i in features]
            distances = sum(distances) / len(features)

            # Get scene count
            scene_count = count_scenes(distances)

            # Calculate predictions
            indices = np.argsort(data['scene_transition_boundary_prediction']).tolist()[::-1]
            pred = sorted(data['scene_transition_boundary_prediction'], reverse=True)
            pred[scene_count:] = [elem / (2 * pred[scene_count] + 1) for elem in pred]
            data['scene_transition_boundary_prediction'] = torch.tensor(
                [x for _, x in sorted(zip(indices, pred), key=lambda pair: pair[0])])

            filename = out_dir + '\\' + os.path.basename(file)
            with open(filename, 'wb') as out_file:
                pickle.dump(data, out_file)
    print(f'Done! check {out_dir} directory for output files')
