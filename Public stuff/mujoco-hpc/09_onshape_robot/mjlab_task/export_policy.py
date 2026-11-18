"""Turn a trained mjlab checkpoint (model_999.pt) into a small file of plain numbers, walk_policy.npz.

    python export_policy.py model_999.pt            # writes walk_policy.npz next to it

Run it anywhere PyTorch is installed (WSL, the cluster). The robot runner (run_on_robot.py) then needs only numpy,
so it can run on your Windows laptop, which is where the Bluetooth to the LEGO Double Motor works.

The file holds the network (20 inputs for Walk-Straight, 11 for Walk-Straight-Real (old) or 18 for the IMU version, then 128 -> 128 -> 2 outputs, ELU)
and the input scaling the policy was trained with.
"""
import sys

import numpy as np
import torch

if len(sys.argv) != 2:
  sys.exit("usage: python export_policy.py model_999.pt")

actor = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["actor_state_dict"]
w = {k: v.numpy() for k, v in actor.items()}
np.savez(
  "walk_policy.npz",
  mean=w["obs_normalizer._mean"][0], std=w["obs_normalizer._std"][0], eps=np.float32(1e-2),
  w0=w["mlp.0.weight"], b0=w["mlp.0.bias"],
  w1=w["mlp.2.weight"], b1=w["mlp.2.bias"],
  w2=w["mlp.4.weight"], b2=w["mlp.4.bias"],
)
print("wrote walk_policy.npz:", w["mlp.0.weight"].shape[1], "inputs ->", w["mlp.0.weight"].shape[0], "->", w["mlp.2.weight"].shape[0], "->", w["mlp.4.weight"].shape[0], "outputs")
