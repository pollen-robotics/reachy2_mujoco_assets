"""
MuJoCo WebSocket Server (Real-time + Camera Support)

 - Real-time simulation (syncs MuJoCo time to wall-clock).
 - WebSocket JSON API for control and state.
 - Multi-camera support with RGB and depth rendering.

Usage:
  python reachy2_mujoco_server.py --model robot.xml --host 0.0.0.0 --port 8765
"""

import argparse
import asyncio
import io
import json
import time
from typing import Any, Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np
import websockets
from PIL import Image
import cv2

# ---------- DEFAULT CONFIG ----------
DEFAULT_STATE_BROADCAST_HZ = 60.0
DEFAULT_SIM_CONTROL_HZ = 1000.0  # Simulation control loop rate
DEFAULT_IMAGE_FPS = 15.0
JPEG_QUALITY = 70

# Shared camera resolution (matching your C++ approach)
SHARED_CAMERA_WIDTH = 320
SHARED_CAMERA_HEIGHT = 240
SHARED_CAMERA_FOVY = 45.0

# Camera configuration with topic paths only (all use shared resolution)
CAMERA_CONFIGS = {
    "left_camera": {
        "topic_paths": {
            "image": "teleop_camera/left_image/image_raw/compressed",
            "camera_info": "teleop_camera/left_image/image_raw/camera_info",
        },
        "has_depth": False,
    },
    "right_camera": {
        "topic_paths": {
            "image": "teleop_camera/right_image/image_raw/compressed",
            "camera_info": "teleop_camera/right_image/image_raw/camera_info",
        },
        "has_depth": False,
    },
    "depth_cam_rgb": {
        "topic_paths": {
            "image": "camera/color/image_raw/compressed",
           "camera_info": "camera/color/image_raw/camera_info", 
            "depth_image": "camera/depth/image_raw",
            "depth_camera_info": "camera/depth/camera_info",
        },
        "has_depth": True,
    },
}
# -----------------------------------


def nd_to_list(a):
    """Convert numpy arrays and numpy types to JSON-serializable types."""
    if hasattr(a, "tolist"):
        return a.tolist()
    elif isinstance(a, (np.integer, np.int32, np.int64)):
        return int(a)
    elif isinstance(a, (np.floating, np.float32, np.float64)):
        return float(a)
    elif isinstance(a, np.ndarray):
        return a.tolist()
    else:
        return a


class CameraRenderer:
    """Handles camera rendering using shared renderer approach."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.cameras = {}  # camera_name -> dict with renderer, width, height, etc.
        self.setup_cameras()

    def setup_cameras(self):
        """Create a renderer per camera and record metadata."""
        # Find available cameras in the model
        for cam_name in CAMERA_CONFIGS.keys():
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id == -1:
                print(f"Warning: Camera '{cam_name}' not found in model, skipping")
                continue

            # Get camera resolution from model.cam_resolution (2 ints per camera)
            # cam_resolution is an int array: [w0, h0, w1, h1, ...]
            cam_res_idx = 2 * cam_id
            try:
                cam_width = int(self.model.cam_resolution[cam_res_idx])
                cam_height = int(self.model.cam_resolution[cam_res_idx + 1])
            except Exception:
                # fallback to shared resolution if not available
                cam_width = SHARED_CAMERA_WIDTH
                cam_height = SHARED_CAMERA_HEIGHT

            # Create a single renderer for this camera
            try:
                renderer = mujoco.Renderer(self.model, cam_width, cam_height)
                print(f"Created renderer for camera '{cam_name}': {cam_width}x{cam_height}")
                    
            except Exception as e:
                print(f"Failed to initialize renderer for camera '{cam_name}': {e}")
                continue

            self.cameras[cam_name] = {
                "id": cam_id,
                "name": cam_name,
                "renderer": renderer,
                "width": cam_width,
                "height": cam_height,
                "fovy": float(self.model.cam_fovy[cam_id]) if hasattr(self.model, "cam_fovy") else SHARED_CAMERA_FOVY,
                "has_depth": CAMERA_CONFIGS.get(cam_name, {}).get("has_depth", False),
                "topic_paths": CAMERA_CONFIGS.get(cam_name, {}).get("topic_paths", {}),
            }

    def render_camera(self, camera_name: str) -> Optional[Dict[str, Any]]:
        """Render a specific camera and return frame data (RGB jpeg bytes, optional depth)."""
        cam = self.cameras.get(camera_name)
        if not cam:
            return None

        width = cam["width"]
        height = cam["height"]
        cam_id = cam["id"]
        renderer = cam["renderer"]

        try:
            # First render RGB
            renderer.update_scene(self.data, camera=camera_name)
            rgb = renderer.render()

            # Process RGB image
            if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

            # MuJoCo often produces images bottom-up: flip vertically
            if rgb.shape[0] == height:
                rgb = np.flipud(rgb)
            else:
                # if sizes mismatch, try to resize to declared width/height
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

            # If RGBA, drop alpha
            if rgb.ndim == 3 and rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]

            # Ensure RGB -> BGR for OpenCV imencode
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Encode to JPEG (like C++), quality near 90
            ok, jpeg_buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                print(f"Warning: JPEG encoding failed for camera {camera_name}")
                return None
            jpeg_bytes = jpeg_buf.tobytes()

            frame_data = {
                "name": camera_name,
                "width": width,
                "height": height,
                "fovy": cam["fovy"],
                "frame_name": f"{camera_name}_optical_frame",
                "topic_paths": cam["topic_paths"],
                "rgb_data": jpeg_bytes,
                "has_depth": cam["has_depth"],
            }

            # Depth rendering - simple approach that was working
            if cam["has_depth"]:
                try:
                    # Enable depth, re-render, then disable 
                    renderer.enable_depth_rendering()
                    renderer.update_scene(self.data, camera=camera_name)
                    depth_raw = renderer.render()
                    renderer.disable_depth_rendering()
                    
                    # Handle depth buffer dimensions (fix the dimension swap that solved the mosaic)
                    if depth_raw.ndim == 3:
                        depth_2d = depth_raw[:, :, 0]
                    else:
                        depth_2d = depth_raw
                    
                    # Fix dimension order if needed (this was the key fix)
                    if depth_2d.shape != (height, width):
                        depth_2d = cv2.resize(depth_2d, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    # Simple conversion - no complex scaling that broke it
                    depth_clean = np.ascontiguousarray(depth_2d, dtype=np.float32)
                    depth_bytes = depth_clean.tobytes()
                    
                    frame_data["depth_data"] = depth_bytes
                    frame_data["depth_encoding"] = "32FC1"
                    
                except Exception as e:
                    print(f"Warning: Failed to render depth for camera '{camera_name}': {e}")
                    frame_data["has_depth"] = False

            return frame_data

        except Exception as e:
            print(f"Error rendering camera '{camera_name}': {e}")
            return None

    def get_available_cameras(self) -> List[str]:
        return list(self.cameras.keys())

    def cleanup(self):
        for cam in self.cameras.values():
            try:
                if cam["renderer"]:
                    cam["renderer"].close()
            except:
                pass


class MujocoServer:
    def __init__(
        self,
        model_path: str,
        enable_viewer: bool = True,
        headless: bool = False,
        state_hz: float = DEFAULT_STATE_BROADCAST_HZ,
        sim_hz: float = DEFAULT_SIM_CONTROL_HZ,
        image_fps: float = DEFAULT_IMAGE_FPS,
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Viewer
        self.enable_viewer = enable_viewer and not headless
        if self.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        # Camera system
        self.headless = headless
        if not headless:
            try:
                self.camera_renderer = CameraRenderer(self.model, self.data)
                self.can_render_cameras = True
                print(
                    f"Camera system initialized with {len(self.camera_renderer.get_available_cameras())} cameras"
                )
            except Exception as e:
                print("Camera system unavailable:", e)
                self.camera_renderer = None
                self.can_render_cameras = False
        else:
            self.camera_renderer = None
            self.can_render_cameras = False

        self.clients = set()
        self._stop = False

        # Rate control
        self.state_hz = state_hz
        self.sim_hz = sim_hz
        self.image_fps = image_fps

        # Control arrays
        self.position_targets = {}  # joint_name -> position
        self.velocity_targets = {}  # joint_name -> velocity
        self.effort_targets = {}  # joint_name -> effort

        # Control modes for each joint
        self.control_modes = {}  # joint_name -> "position", "velocity", or "effort"

        # PID gains for position control (simple implementation)
        self.position_kp = 100.0
        self.position_kd = 10.0

        # Time sync
        self.start_wall = time.time()
        self.start_sim = self.data.time

        # Print configuration
        print(f"Server configuration:")
        print(f"  - Simulation rate: {self.sim_hz} Hz")
        print(f"  - State broadcast rate: {self.state_hz} Hz")
        print(f"  - Image streaming rate: {self.image_fps} FPS")
        print(f"  - Number of joints: {self.model.njnt}")
        print(f"  - Number of actuators: {self.model.nu}")
        if self.can_render_cameras:
            print(
                f"  - Available cameras: {', '.join(self.camera_renderer.get_available_cameras())}"
            )

    def apply_controls(self):
        """Apply control commands based on joint names and control modes."""
        # Clear control array
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0

        for joint_name, mode in self.control_modes.items():
            # Find joint and actuator indices
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id == -1:
                continue

            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
            )
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qvel_adr = self.model.jnt_dofadr[joint_id]

            if mode == "position" and joint_name in self.position_targets:
                target = self.position_targets[joint_name]
                if actuator_id != -1:
                    # If there's an actuator, use it for position control
                    self.data.ctrl[actuator_id] = target
                else:
                    # Use PID control via qfrc_applied
                    error = target - self.data.qpos[qpos_adr]
                    vel = self.data.qvel[qvel_adr]
                    force = self.position_kp * error - self.position_kd * vel
                    self.data.qfrc_applied[qvel_adr] = force

            elif mode == "velocity" and joint_name in self.velocity_targets:
                target = self.velocity_targets[joint_name]
                if actuator_id != -1:
                    self.data.ctrl[actuator_id] = target

            elif mode == "effort" and joint_name in self.effort_targets:
                target = self.effort_targets[joint_name]
                self.data.qfrc_applied[qvel_adr] = target

    def step_realtime(self):
        """Step simulation so that sim time follows wall-clock."""
        # target sim time
        elapsed_wall = time.time() - self.start_wall
        target_time = self.start_sim + elapsed_wall
        if self.data.time < target_time:
            mujoco.mj_step(self.model, self.data)
            # Update viewer if enabled
            if self.enable_viewer and self.viewer is not None:
                self.viewer.sync()
        else:
            # sim caught up, sleep a little
            time.sleep(0.001)

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": float(self.data.time),
            "qpos": nd_to_list(self.data.qpos),
            "qvel": nd_to_list(self.data.qvel),
            "ctrl": nd_to_list(self.data.ctrl),
            "qfrc_applied": nd_to_list(self.data.qfrc_applied),
            "sensordata": nd_to_list(self.data.sensordata),
        }

    async def handle_message(self, msg: str, ws):
        try:
            obj = json.loads(msg)
        except Exception as e:
            print(f"JSON parse error: {e}")
            try:
                await ws.send(json.dumps({"type": "error", "reason": "invalid_json"}))
            except:
                pass  # Client might have disconnected
            return

        t = obj.get("type")
        try:
            if t == "set_joint_control":
                # Handle joint-level control commands
                joint_name = obj.get("joint_name")
                mode = obj.get("mode")  # "position", "velocity", or "effort"
                value = float(obj.get("value", 0.0))  # Ensure it's a float

                if joint_name:
                    self.control_modes[joint_name] = mode
                    if mode == "position":
                        self.position_targets[joint_name] = value
                    elif mode == "velocity":
                        self.velocity_targets[joint_name] = value
                    elif mode == "effort":
                        self.effort_targets[joint_name] = value

                    await ws.send(
                        json.dumps(
                            {
                                "type": "ack",
                                "what": "set_joint_control",
                                "joint": joint_name,
                                "mode": mode,
                            }
                        )
                    )

            elif t == "set_bulk_control":
                # Handle bulk control commands (all joints at once)
                joints = obj.get("joints", [])
                for joint_data in joints:
                    joint_name = joint_data.get("name")
                    mode = joint_data.get("mode")
                    value = float(joint_data.get("value", 0.0))  # Ensure it's a float

                    if joint_name:
                        self.control_modes[joint_name] = mode
                        if mode == "position":
                            self.position_targets[joint_name] = value
                        elif mode == "velocity":
                            self.velocity_targets[joint_name] = value
                        elif mode == "effort":
                            self.effort_targets[joint_name] = value

                await ws.send(json.dumps({"type": "ack", "what": "set_bulk_control"}))

            elif t == "reset":
                st = obj.get("state", {})
                if "qpos" in st:
                    # Convert to list and ensure proper types
                    qpos_values = st["qpos"]
                    for i in range(min(len(qpos_values), self.model.nq)):
                        self.data.qpos[i] = float(qpos_values[i])
                if "qvel" in st:
                    # Convert to list and ensure proper types
                    qvel_values = st["qvel"]
                    for i in range(min(len(qvel_values), self.model.nv)):
                        self.data.qvel[i] = float(qvel_values[i])
                mujoco.mj_forward(self.model, self.data)

                # Clear control targets
                self.position_targets.clear()
                self.velocity_targets.clear()
                self.effort_targets.clear()
                self.control_modes.clear()

                await ws.send(json.dumps({"type": "ack", "what": "reset"}))

            elif t == "get_model_info":
                # Provide model information to client
                joint_info = []
                for i in range(self.model.njnt):
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if name:
                        joint_info.append(
                            {
                                "name": name,
                                "qpos_adr": int(
                                    self.model.jnt_qposadr[i]
                                ),  # Convert to int
                                "qvel_adr": int(
                                    self.model.jnt_dofadr[i]
                                ),  # Convert to int
                                "type": int(self.model.jnt_type[i]),  # Convert to int
                            }
                        )

                # Add camera information
                camera_info = []
                if self.can_render_cameras:
                    for cam_name in self.camera_renderer.get_available_cameras():
                        cam_data = self.camera_renderer.cameras[cam_name]
                        config = CAMERA_CONFIGS.get(cam_name, {})
                        camera_info.append(
                            {
                                "name": cam_name,
                                "width": cam_data["width"],
                                "height": cam_data["height"],
                                "fovy": cam_data["fovy"],
                                "frame_name": f"{cam_name}_optical_frame",
                                "has_depth": config.get("has_depth", False),
                                "topic_paths": config.get("topic_paths", {}),
                            }
                        )

                await ws.send(
                    json.dumps(
                        {
                            "type": "model_info",
                            "joints": joint_info,
                            "cameras": camera_info,
                            "njnt": int(self.model.njnt),  # Convert to int
                            "nq": int(self.model.nq),  # Convert to int
                            "nv": int(self.model.nv),  # Convert to int
                            "nu": int(self.model.nu),  # Convert to int
                        }
                    )
                )

            elif t == "get_available_cameras":
                # Return list of available cameras
                cameras = []
                if self.can_render_cameras:
                    cameras = self.camera_renderer.get_available_cameras()

                await ws.send(
                    json.dumps({"type": "available_cameras", "cameras": cameras})
                )

            else:
                await ws.send(
                    json.dumps({"type": "error", "reason": f"unknown_command: {t}"})
                )
        except Exception as e:
            print(f"Error handling message '{t}' from client: {e}")
            import traceback

            traceback.print_exc()

    async def ws_handler(self, ws):
        self.clients.add(ws)
        print(f"Client connected (total: {len(self.clients)})")
        try:
            async for msg in ws:
                await self.handle_message(msg, ws)
        except Exception as e:
            print(f"WebSocket handler error: {e}")
        finally:
            self.clients.discard(ws)
            print(f"Client disconnected (remaining: {len(self.clients)})")

    async def state_loop(self):
        interval = 1.0 / self.state_hz
        img_interval = 1.0 / self.image_fps if self.image_fps > 0 else None
        last_img = 0.0

        while not self._stop:
            t0 = time.time()
            state = self.get_state()
            payload = json.dumps({"type": "state", "state": state})

            # Send state to all clients, handling disconnections gracefully
            disconnected_clients = set()
            for client in self.clients.copy():
                try:
                    await client.send(payload)
                except Exception as e:
                    print(f"Client disconnected during state send: {e}")
                    disconnected_clients.add(client)

            # Remove disconnected clients
            for client in disconnected_clients:
                self.clients.discard(client)

            # Handle camera rendering and streaming
            if img_interval and (time.time() - last_img >= img_interval):
                await self.stream_camera_frames()
                last_img = time.time()

            await asyncio.sleep(max(0, interval - (time.time() - t0)))

    async def stream_camera_frames(self):
        """Stream frames from all available cameras."""
        if not self.can_render_cameras:
            return

        disconnected_clients = set()

        # Render and stream each camera
        for camera_name in self.camera_renderer.get_available_cameras():
            frame_data = self.camera_renderer.render_camera(camera_name)
            if not frame_data:
                continue

            # Create camera frame header (exactly matching C++ expectations)
            header = {
                "type": "camera_frame",
                "camera_name": frame_data["name"],
                "width": frame_data["width"],
                "height": frame_data["height"],
                "fovy": frame_data["fovy"],
                "frame_name": frame_data["frame_name"],
                "topic_paths": frame_data["topic_paths"],
                "has_depth": frame_data["has_depth"],
                "rgb_size": len(frame_data["rgb_data"]),
            }

            if frame_data["has_depth"] and "depth_data" in frame_data:
                header["depth_size"] = len(frame_data["depth_data"])
                header["depth_encoding"] = frame_data["depth_encoding"]

            header_json = json.dumps(header)

            # Send to all clients (matching C++ client expectations exactly)
            for client in self.clients.copy():
                try:
                    # Send header as text message
                    await client.send(header_json)

                    # Send RGB data as binary message
                    await client.send(frame_data["rgb_data"])

                    # Send depth data as binary message if available
                    if frame_data["has_depth"] and "depth_data" in frame_data:
                        await client.send(frame_data["depth_data"])

                except Exception as e:
                    print(f"Client disconnected during camera frame send: {e}")
                    disconnected_clients.add(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)

    async def sim_loop(self):
        sim_interval = 1.0 / self.sim_hz
        while not self._stop:
            t0 = time.time()
            self.apply_controls()
            mujoco.mj_step(self.model, self.data)

            # Update viewer if enabled
            if self.enable_viewer and self.viewer is not None:
                self.viewer.sync()

            # Sleep to maintain simulation rate
            elapsed = time.time() - t0
            sleep_time = max(0, sim_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def run(self, host, port):
        server = await websockets.serve(self.ws_handler, host, port, max_size=None)
        print(f"Server running on ws://{host}:{port}")
        tasks = [
            asyncio.create_task(self.sim_loop()),
            asyncio.create_task(self.state_loop()),
        ]
        await server.wait_closed()
        self._stop = True
        for t in tasks:
            t.cancel()

    def cleanup(self):
        """Cleanup resources."""
        if self.camera_renderer:
            self.camera_renderer.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to MuJoCo model XML file")
    ap.add_argument("--host", default="127.0.0.1", help="Server host address")
    ap.add_argument("--port", type=int, default=8765, help="Server port")
    ap.add_argument(
        "--no-viewer", action="store_true", help="Disable the MuJoCo viewer"
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no viewer, no renderer)",
    )

    # Rate control arguments
    ap.add_argument(
        "--state-hz",
        type=float,
        default=DEFAULT_STATE_BROADCAST_HZ,
        help=f"State broadcast rate in Hz (default: {DEFAULT_STATE_BROADCAST_HZ})",
    )
    ap.add_argument(
        "--sim-hz",
        type=float,
        default=DEFAULT_SIM_CONTROL_HZ,
        help=f"Simulation control rate in Hz (default: {DEFAULT_SIM_CONTROL_HZ})",
    )
    ap.add_argument(
        "--image-fps",
        type=float,
        default=DEFAULT_IMAGE_FPS,
        help=f"Image streaming rate in FPS (default: {DEFAULT_IMAGE_FPS})",
    )

    args = ap.parse_args()

    srv = MujocoServer(
        args.model,
        enable_viewer=not args.no_viewer,
        headless=args.headless,
        state_hz=args.state_hz,
        sim_hz=args.sim_hz,
        image_fps=args.image_fps,
    )
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(srv.run(args.host, args.port))
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        srv.cleanup()
        loop.stop()


if __name__ == "__main__":
    main()