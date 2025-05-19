"""
    Use AI2-THOR to collect RGB, depth, instance masks, and object metadata per scene.
"""
import os
import numpy as np
import cv2
from ai2thor.controller import Controller
from tqdm import tqdm

import os
import numpy as np
import cv2
from ai2thor.controller import Controller
from tqdm import tqdm

def generate_light_data(
    base_dir="data/light_test",
    scenes=["FloorPlan1", "FloorPlan2"],  # No _physics scenes!
    grid_size=0.5,
    rotate_step_degrees=90,
    max_steps_per_scene=15
):
    os.makedirs(base_dir, exist_ok=True)

    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        gridSize=grid_size,
        rotateStepDegrees=rotate_step_degrees,
        fieldOfView=90,
        width=300,
        height=300,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        snapToGrid=True
    )

    sample_count = 0

    for scene in scenes:
        print(f"🔄 Generating data for {scene}")
        controller.reset(scene)

        try:
            event = controller.step("GetReachablePositions")
            reachable_positions = event.metadata["actionReturn"]
        except Exception as e:
            print(f"❌ Failed to get reachable positions in {scene}: {e}")
            continue

        np.random.shuffle(reachable_positions)
        positions = reachable_positions[:max_steps_per_scene]

        for pos in tqdm(positions):
            for rot in [0, 90, 180, 270]:
                try:
                    event = controller.step(
                        action="TeleportFull",
                        x=pos["x"],
                        y=pos["y"],
                        z=pos["z"],
                        rotation={"x": 0, "y": rot, "z": 0},
                        horizon=0.0,
                        standing=True
                    )

                    rgb = event.frame
                    depth = event.depth_frame
                    instance = event.instance_segmentation_frame

                    rgb_path = os.path.join(base_dir, f"rgb_{sample_count:05d}.png")
                    depth_path = os.path.join(base_dir, f"depth_{sample_count:05d}.npy")
                    inst_path = os.path.join(base_dir, f"instance_{sample_count:05d}.png")

                    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    np.save(depth_path, depth)
                    cv2.imwrite(inst_path, instance)

                    sample_count += 1

                except Exception as e:
                    print(f"⚠️ Skipped sample at {pos} rotation {rot}: {e}")
                    continue

    controller.stop()
    print(f"✅ Done. {sample_count} samples saved in {base_dir}")

if __name__ == "__main__":
    generate_light_data()