#### Let's build AI-powered software based on SOTA models and architecture and try to capture full potential of these interesting innovations.


### Features:
- PointNet++ for 3D object detection and scene segmentation
- using [AI2THOR](https://ai2thor.allenai.org) as indoor data generation.
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE) Model for Vision-and-Language Navigation in Continuous Environments


### Future:
- command-based navigation system using (Matterport3D)[https://niessner.github.io/Matterport/#download] for Vision-and-Language Navigation in Continuous Environments(VLN-CE)
- generate physics-based data using [AI2THOR](https://ai2thor.allenai.org) and integrate physics-informed models
- implementing VLN_CE model using [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- implement [PointNet++](https://github.com/fxia22/pointnet2) for 3D object detection and scene segmentation using [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)


### Ref:
- [PointNet++](https://github.com/fxia22/pointnet2) for 3D object detection and scene segmentation
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE) for Vision-and-Language Navigation in Continuous Environments
- [Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments](https://arxiv.org/pdf/2004.02857) for VLN-CE
- [AI2THOR](https://ai2thor.allenai.org) for indoor data generation
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) for indoor data generation


##### Usage:

- to start training PointNet++ for 3D object detection and scene segmentation on AI2THOR dataset:
```bash
python3 script/train.py --conifg path/to/configs.json --model PointNetPP
```

- to start training VLN-CE for Vision-and-Language Navigation in Continuous Environments on [VLN-CE](https://jacobkrantz.github.io/vlnce/data) dataset:
```bash
python3 script/VLN_CE/main.py
```