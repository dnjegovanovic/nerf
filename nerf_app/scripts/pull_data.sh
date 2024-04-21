if ! [ -f /d/ML_AI_DL_Projects/projects_repo/nerf/data/tiny_nerf_data.npz ]; then
  echo "Downloading file ... "
  curl --url http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz \
       --output /d/ML_AI_DL_Projects/projects_repo/nerf/data/tiny_nerf_data.npz
fi