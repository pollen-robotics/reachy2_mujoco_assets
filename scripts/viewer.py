# scripts/viewer.py

import os
import sys
import mujoco
import mujoco.viewer
import argparse


def load_model(xml_path: str):
    if not os.path.exists(xml_path):
        print(f"[ERROR] File not found: {xml_path}")
        sys.exit(1)
    return mujoco.MjModel.from_xml_path(xml_path)


def main(xml_path):
    print(f"[INFO] Loading model: {xml_path}")
    model = load_model(xml_path)
    data = mujoco.MjData(model)

    # Launch the interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[INFO] Viewer launched. Close the window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
    print("[INFO] Viewer closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a Mujoco XML model")
    parser.add_argument(
        "xml_path",
        type=str,
        help="Path to the Mujoco XML file (robot or scene model)",
    )
    args = parser.parse_args()

    main(args.xml_path)
