#### Let's build AI-powered software based on SOTA models and architecture and try to capture full potential of these interesting innovations.


### Features:
- [x] PointNet++ for 3D object detection and scene segmentation
- [x] using [AI2THOR](https://ai2thor.allenai.org) as indoor data generation.
- [x] [VLN-CE](https://github.com/jacobkrantz/VLN-CE) Model for Vision-and-Language Navigation in Continuous Environments
- [x] command-based navigation system using (Matterport3D)[https://niessner.github.io/Matterport/#download] for Vision-and-Language Navigation in Continuous Environments(VLN-CE)

- Improvement specific to VLN-CE:
    - [x] Using a custom environment for VLN-CE that implements the *Gymnasium* interface. This provides similar functionality to *Habitat* but with a simpler implementation.
    - [x] Replace simple cross-attention with **cross-modal transformers** like those in ViLBERT
    


### Future:

- [ ] generate physics-based data using [AI2THOR](https://ai2thor.allenai.org) and integrate physics-informed models
- [ ] implementing VLN_CE model using [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- [ ] implement [PointNet++](https://github.com/fxia22/pointnet2) for 3D object detection and scene segmentation using [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)

- Specific to VLN-CE:

    - [ ] integrate [Language-Aligned Waypoint (LAW) Supervision for Vision-and-Language Navigation in Continuous Environments](https://3dlg-hcvc.github.io/LAW-VLNCE/?utm_source=chatgpt.com) into VLN-CE model
    - [ ] integrate [EnvEdit: Environment Editing for Vision-and-Language Navigation](https://arxiv.org/pdf/2203.15685) into VLN-CE model for data augmentation
    - [ ] integrate [VLN-PETL: Parameter-Efficient Transfer Learning for Vision-and-Language Navigation](https://arxiv.org/pdf/2308.10172) for reducing computational costs
    - [ ] Use BERT, RoBERTa, or DistilBERT for language understanding.
    - [ ] Use ViT or CLIP-ViT for visual encoding to better capture object-level semantics.
    - [ ]



### Ref:
- [PointNet++](https://github.com/fxia22/pointnet2) for 3D object detection and scene segmentation
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE) for Vision-and-Language Navigation in Continuous Environments
- [Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments](https://arxiv.org/pdf/2004.02857) for VLN-CE
- [AI2THOR](https://ai2thor.allenai.org) for indoor data generation
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) for indoor data generation


### Tracking VLN-CE loss after each improvement for just 30 episodes:
- changing simple CorssModalAttention to ViLBERT and CrossModalTransformer improved the generalization by $18.1%$
- improving **memory management**, optimization strategy and tuning learning rate and initialization plus using pre-trained encoders like **DistilBERT**, **ViT encoder** and **MobileNet** lang-encoder dropped loss by $98%$ and lead to performance prior to last stage in contrast with being more computationally expensive. memory management is more efficient and also leads to overfit. data augmentation is needed for better data and it's gonna be the next stage to tune dataset for more diverse set of instructions and pathways.
- 


##### Usage:

- to start training PointNet++ for 3D object detection and scene segmentation on AI2THOR dataset:
```bash
python3 script/train.py --conifg configs/PointNetPP.json --model PointNetPP
```

- to start training VLN-CE for Vision-and-Language Navigation in Continuous Environments on [VLN-CE](https://jacobkrantz.github.io/vlnce/data) dataset:
```bash
python3 script/VLN_CE/main.py
```