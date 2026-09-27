# RoboMaster SLAM + Sign Detection

This project runs the Grid SLAM/DFS implementation from the `SLAM` submodule
while detecting colored signs from the RoboMaster camera. Confirmed signs are
saved in the current grid cell and wall direction in the exported map. It does
not navigate to or shoot signs.

## Clone

Clone with submodules so both the original SLAM code and the RoboMaster SDK
source needed for the camera codec are present:

```powershell
git clone --recurse-submodules https://github.com/punkrub/SLAM-DETECT.git
cd SLAM-DETECT
```

If the repository was cloned without submodules, initialize them with:

```powershell
git submodule update --init --recursive
```

## Install (Windows, Python 3.8)

The RoboMaster camera stream requires the SDK's native `libmedia_codec` module.
Building it on Windows requires CMake and Visual Studio C++ Build Tools.

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
Push-Location .\RoboMaster-SDK
python setup_with_lib.py install
Pop-Location
```

## Run

Connect the computer to the RoboMaster access point, then run:

```powershell
python slam_detect_camera.py --conn-type ap
```

Press `q` in the camera window to stop. Each run creates a new folder under
`mission_maps/` containing `explored_map.json` and `map.png`. The image includes
the explored walls, estimated path, start/end positions, and marked signs.

For a SLAM simulation with a PC webcam instead of the robot camera:

```powershell
python slam_detect_camera.py --mock
```

## Included files

- `slam_detect_camera.py`: coordinates SLAM exploration, camera inspections,
  sign marking, and per-run map output.
- `detect_camera.py`: color/shape detector used by the integrated runner.
- `SLAM`: pinned submodule; the upstream SLAM source is not modified here.
- `RoboMaster-SDK`: upstream SDK submodule used to build the native camera codec.

`detect.py` is not required by this project. `detect_camera.py` defines the
detector colors it uses directly.