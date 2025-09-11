"""
MuJoCo WebSocket Server (Real-time + Binary Images)

 - Real-time simulation (syncs MuJoCo time to wall-clock).
 - WebSocket JSON API for control and state.
 - Binary JPEG image streaming over WebSocket.

Usage:
  python mujoco_ws_server.py --model robot.xml --host 0.0.0.0 --port 8765
"""

import argparse
import asyncio
import io
import json
import time
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
import websockets
from PIL import Image

# ---------- DEFAULT CONFIG ----------
DEFAULT_STATE_BROADCAST_HZ = 60.0
DEFAULT_SIM_CONTROL_HZ = 1000.0  # Simulation control loop rate
DEFAULT_IMAGE_FPS = 15.0
JPEG_QUALITY = 70
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

        # Renderer (offscreen)
        self.img_w = 320
        self.img_h = 240
        self.headless = headless
        if not headless:
            try:
                self.renderer = mujoco.Renderer(self.model, self.img_w, self.img_h)
                self.can_render = True
            except Exception as e:
                print("Renderer unavailable:", e)
                self.renderer = None
                self.can_render = False
        else:
            self.renderer = None
            self.can_render = False

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

    def render_jpeg(self) -> Optional[bytes]:
        if not self.can_render:
            return None
        self.renderer.update_scene(self.data)
        rgb = self.renderer.render()
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()

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

                await ws.send(
                    json.dumps(
                        {
                            "type": "model_info",
                            "joints": joint_info,
                            "njnt": int(self.model.njnt),  # Convert to int
                            "nq": int(self.model.nq),  # Convert to int
                            "nv": int(self.model.nv),  # Convert to int
                            "nu": int(self.model.nu),  # Convert to int
                        }
                    )
                )

            else:
                await ws.send(
                    json.dumps({"type": "error", "reason": f"unknown_command: {t}"})
                )
        except Exception as e:
            print(f"Error handling message '{t}' from client: {e}")
            import traceback

            traceback.print_exc()

    async def ws_handler(self, ws, path):
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

            if img_interval and (time.time() - last_img >= img_interval):
                jpeg = self.render_jpeg()
                if jpeg:
                    header = json.dumps(
                        {
                            "type": "image_jpeg",
                            "width": self.img_w,
                            "height": self.img_h,
                            "size": len(jpeg),
                        }
                    )
                    disconnected_clients = set()
                    for client in self.clients.copy():
                        try:
                            # send header as text, then image as binary
                            await client.send(header)
                            await client.send(jpeg)  # binary frame
                        except Exception as e:
                            print(f"Client disconnected during image send: {e}")
                            disconnected_clients.add(client)

                    # Remove disconnected clients
                    for client in disconnected_clients:
                        self.clients.discard(client)

                last_img = time.time()

            await asyncio.sleep(max(0, interval - (time.time() - t0)))

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
        loop.stop()


if __name__ == "__main__":
    main()
