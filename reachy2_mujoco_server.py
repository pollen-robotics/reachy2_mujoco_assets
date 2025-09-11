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
    return a.tolist() if hasattr(a, "tolist") else a


class MujocoServer:
    def __init__(self, model_path: str, enable_viewer: bool = True, headless: bool = False, 
                 state_hz: float = DEFAULT_STATE_BROADCAST_HZ, 
                 sim_hz: float = DEFAULT_SIM_CONTROL_HZ,
                 image_fps: float = DEFAULT_IMAGE_FPS):
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

        # Control
        self.control_mode = "torque"
        self.torque_cmd = np.zeros(self.model.nu)
        self.position_target = np.zeros(self.model.nq)

        # Time sync
        self.start_wall = time.time()
        self.start_sim = self.data.time
        
        # Print configuration
        print(f"Server configuration:")
        print(f"  - Simulation rate: {self.sim_hz} Hz")
        print(f"  - State broadcast rate: {self.state_hz} Hz") 
        print(f"  - Image streaming rate: {self.image_fps} FPS")

    def apply_controls(self):
        if self.control_mode == "torque":
            self.data.ctrl[:] = self.torque_cmd[: self.model.nu]
        elif self.control_mode == "position":
            Kp, Kd = 100.0, 2.0
            err = self.position_target - self.data.qpos[: self.model.nq]
            vel = self.data.qvel[: self.model.nv]
            tau = Kp * err - Kd * vel
            self.data.qfrc_applied[: tau.size] = tau

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
        except Exception:
            try:
                await ws.send(json.dumps({"type": "error", "reason": "invalid_json"}))
            except:
                pass  # Client might have disconnected
            return

        t = obj.get("type")
        try:
            if t == "set_mode":
                self.control_mode = obj.get("mode", "torque")
                await ws.send(json.dumps({"type": "ack", "mode": self.control_mode}))
            elif t == "set_torque":
                arr = np.array(obj.get("torque", []), dtype=float)
                self.torque_cmd[: self.model.nu] = np.resize(arr, self.model.nu)
                await ws.send(json.dumps({"type": "ack", "what": "set_torque"}))
            elif t == "set_position_target":
                arr = np.array(obj.get("position", []), dtype=float)
                self.position_target[: self.model.nq] = np.resize(arr, self.model.nq)
                await ws.send(json.dumps({"type": "ack", "what": "set_position_target"}))
            elif t == "reset":
                st = obj.get("state", {})
                if "qpos" in st:
                    self.data.qpos[: len(st["qpos"])] = st["qpos"]
                if "qvel" in st:
                    self.data.qvel[: len(st["qvel"])] = st["qvel"]
                mujoco.mj_forward(self.model, self.data)
                await ws.send(json.dumps({"type": "ack", "what": "reset"}))
            else:
                await ws.send(json.dumps({"type": "error", "reason": "unknown_command"}))
        except Exception as e:
            print(f"Error handling message from client: {e}")
            # Client might have disconnected, don't try to send response

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
    ap.add_argument("--no-viewer", action="store_true", help="Disable the MuJoCo viewer")
    ap.add_argument("--headless", action="store_true", help="Run in headless mode (no viewer, no renderer)")
    
    # Rate control arguments
    ap.add_argument("--state-hz", type=float, default=DEFAULT_STATE_BROADCAST_HZ, 
                    help=f"State broadcast rate in Hz (default: {DEFAULT_STATE_BROADCAST_HZ})")
    ap.add_argument("--sim-hz", type=float, default=DEFAULT_SIM_CONTROL_HZ,
                    help=f"Simulation control rate in Hz (default: {DEFAULT_SIM_CONTROL_HZ})")
    ap.add_argument("--image-fps", type=float, default=DEFAULT_IMAGE_FPS,
                    help=f"Image streaming rate in FPS (default: {DEFAULT_IMAGE_FPS})")
    
    args = ap.parse_args()

    srv = MujocoServer(args.model, 
                      enable_viewer=not args.no_viewer, 
                      headless=args.headless,
                      state_hz=args.state_hz,
                      sim_hz=args.sim_hz,
                      image_fps=args.image_fps)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(srv.run(args.host, args.port))
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        loop.stop()


if __name__ == "__main__":
    main()
