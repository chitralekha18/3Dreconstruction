from src.resampling import resampling_main
from src.robust_surface_splitting import robust_splitting_main

__author__ = 'Dmitrii'

if __name__ == "__main__":
    # paths where results of splitting will be saved
    # resampling method takes this paths as input
    front_ply_path = "./front_left_surface/front_surface.ply"
    left_ply_path = "./front_left_surface/left_surface.ply"

    # path to results
    path_to_output = "./output/"

    # splitting
    robust_splitting_main(front_ply_path, left_ply_path)
    #resampling

    resampling_main(front_ply_path, left_ply_path, path_to_output)


