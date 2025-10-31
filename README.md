# Galbot

##### Installation of Polymetis #####

***Polymetis is highly environment-dependent, so you must install the specific library versions exactly as required.
Ideally, isolate this environment and have it communicate with other environment that need other libraries via ZMQ.***

--Clone repo
git clone git@github.com:facebookresearch/fairo
cd fairo/polymetis

--Create environment
conda env create -f ./polymetis/environment.yml
conda activate polymetis-local

--Install Python package in editable mode
pip install -e ./polymetis

--Build Frankalib from source
./scripts/build_libfranka.sh <version_tag_or_commit_hash>  **Franka Emika Research 3 needs version 0.15+. By default, the above command-line instructions prioritize binding libfranka to the ROS path.**
./scripts/build_libfranka_conda.sh <version_tag_or_commit_hash> **To build a ROS-free, conda-only dependency setup of Polymetis, you need to run this command.** 

--Build Polymetis from source
mkdir -p ./polymetis/build
cd ./polymetis/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=[OFF/ON] -DBUILD_TESTS=[OFF/ON] -DBUILD_DOCS=[OFF/ON]
make -j

##### Installation of Imitation Learning Environment #####
