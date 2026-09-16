# External model implementations

The model implementations used by the Chapter 4 classification experiments are
kept as Git submodules. GPBSF owns the dataset split, preprocessing, training
budget, evaluation metrics, and result aggregation; the submodules only provide
the neural network definitions.

| Adapter name | Repository | Pinned commit | Role |
| --- | --- | --- | --- |
| `pointnet2` | `beta1scat/Pointnet_Pointnet2_pytorch` | `64d536574ee6a526aca99dbf7de620422893117a` | PointNet++ SSG baseline |
| `mamba3d` | `beta1scat/Mamba3D_for_GPBSF` | `0573a1fe6c78c276d019bff940039909623d59b2` | Proposed topology classifier |

Clone the parent repository with:

```bash
git clone --recurse-submodules <GPBSF repository URL>
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

Do not edit the submodule sources for an experiment. Any compatibility logic
belongs in `experiments/classification/adapters/` so that the upstream commits
remain auditable.

The PointNet++ submodule contains an MIT license. The Mamba3D fork currently
does not expose a license file at its repository root; its redistribution and
publication terms must be confirmed before a public release.
