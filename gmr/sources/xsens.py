"""Xsens MVN real-time mocap client (drop-in replacement for ``NoitomClient``).

Receives Xsens MVN Network Streamer datagrams (default port 9763, MXTP02
pose-quaternion) and exposes the same ``start_thread`` / ``get_frame_data``
/ ``stop`` API as :class:`noitom.NoitomClient` so it slots into the
existing :func:`deploy.retarget._retarget_worker` GMR pipeline without
any algorithmic change.

Protocol summary (MVN real-time network streaming spec, MXTP02):

* 24-byte header (ID string "MXTP02", sample counter, datagram counter,
  ``num_items``, time code, payload size at offset 22..23, ...).
* ``num_items`` body segments follow, each 32 bytes (big-endian):

    4 B  segment_id (1-based, see below)
    12 B position (x, y, z) in METRES
    16 B quaternion (q1=w, q2=x, q3=y, q4=z)

* World frame is **Z-up, right-handed**, origin at the subject's right
  heel.  This already matches the Unitree G1 convention (Z up, X
  forward, Y left) so **no global axis swap is needed** -- contrast
  Noitom which is Y-up and requires the ``[[0,0,1],[1,0,0],[0,1,0]]``
  rotation in ``noitom.get_mocap_noitom_data.quat_fk_noitom``.

* Position is in metres already (Noitom streams cm so it divides by 100;
  here we do nothing).

* The 23 standard body segments are (0-based index = id - 1):
  Pelvis(0), L5(1), L3(2), T12(3), T8(4), Neck(5), Head(6),
  RShoulder(7), RUpperArm(8), RForearm(9), RHand(10),
  LShoulder(11), LUpperArm(12), LForearm(13), LHand(14),
  RUpperLeg(15), RLowerLeg(16), RFoot(17), RToe(18),
  LUpperLeg(19), LLowerLeg(20), LFoot(21), LToe(22).

We expose only the 14 segments referenced by the
``fbx_to_g1_xsens.json`` IK config (Hips, Spine1, [Left|Right] x
[UpLeg|Leg|Foot|Arm|ForeArm|Hand]).  The frame dict therefore looks
exactly like what ``NoitomClient.get_frame_data()`` returns and can be
fed straight into ``GMR.retarget(frame)``.

Caveats:

* The Xsens per-segment LOCAL frame convention is biomechanical and
  differs from Noitom's FBX-style local frames, so a dedicated
  ``fbx_xsens`` GMR IK config is required.  Generate / regenerate it
  with ``python -m deploy.xsens.make_ik_config``.  All Xsens entry
  points (``deploy.play_track --mocap-type xsens``,
  ``deploy.onboard_deploy_wo_GMR.host_sender --mocap-type xsens``,
  ``deploy.xsens.retarget_vis``) load this config unconditionally.

* TCP has no message boundaries: a single ``recv()`` can contain
  multiple packets or a partial packet.  We do proper framing here by
  scanning for the ``MXTP`` magic and using the payload-size field in
  the header.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from queue import Empty, Queue
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------
# Segment-id (0-based) -> GMR human body name expected by
# ``fbx_to_g1_xsens.json``.  Map only what the IK config actually
# references; everything else (head, shoulders, toes) is ignored.
# ----------------------------------------------------------------------
XSENS_SEGMENT_TO_BODY = {
    0:  "Hips",          # Pelvis
    4:  "Spine1",        # T8 (mid-thoracic; best match for FBX Spine1)
    8:  "RightArm",      # Right Upper Arm
    9:  "RightForeArm",  # Right Forearm
    10: "RightHand",     # Right Hand
    12: "LeftArm",       # Left Upper Arm
    13: "LeftForeArm",   # Left Forearm
    14: "LeftHand",      # Left Hand
    15: "RightUpLeg",    # Right Upper Leg
    16: "RightLeg",      # Right Lower Leg
    17: "RightFoot",     # Right Foot
    19: "LeftUpLeg",     # Left Upper Leg
    20: "LeftLeg",       # Left Lower Leg
    21: "LeftFoot",      # Left Foot
}

# The IK config requires these names to all exist in the frame dict.
_REQUIRED_BODIES = set(XSENS_SEGMENT_TO_BODY.values())

# MXTP packet layout constants
_PACKET_MAGIC = b"MXTP"
_HEADER_SIZE = 24
_SEGMENT_SIZE = 32  # 4 (id) + 3*4 (pos) + 4*4 (quat)


class XsensClient:
    """Threaded Xsens MVN network-stream consumer.

    Usage mirrors :class:`noitom.NoitomClient` for drop-in compatibility::

        client = XsensClient(host="0.0.0.0", port=9763, protocol="tcp")
        client.start_thread()
        frame = client.get_frame_data(timeout=True)   # dict[name, (pos, quat)]
        ...
        client.stop()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9763,
        protocol: str = "tcp",
        accept_timeout: float = 60.0,
        queue_max: int = 8,
        require_all_bodies: bool = True,
        verbose: bool = True,
    ):
        self.host = host
        self.port = int(port)
        self.protocol = protocol.lower()
        if self.protocol not in ("tcp", "udp"):
            raise ValueError(f"protocol must be 'tcp' or 'udp', got {protocol!r}")
        self.accept_timeout = float(accept_timeout)
        self.require_all_bodies = bool(require_all_bodies)
        self.verbose = bool(verbose)

        self._queue: Queue = Queue(maxsize=int(queue_max))
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._server_sock: Optional[socket.socket] = None
        self._conn_sock: Optional[socket.socket] = None  # populated only for TCP

        # Diagnostics
        self._frame_count = 0
        self._dropped_partial_packets = 0
        self.fps = 0.0

    # ------------------------------------------------------------------
    # Public API (Noitom-compatible)
    # ------------------------------------------------------------------
    def start_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="XsensClient"
        )
        self._thread.start()

    def is_thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_queue_size(self) -> int:
        return self._queue.qsize()

    def get_frame_data(self, timeout=None):
        """Return the next frame dict ``{body_name: (pos, quat_wxyz)}``.

        Mirrors :meth:`NoitomClient.get_frame_data` semantics:

        * ``timeout=None``  -> non-blocking; ``None`` if empty.
        * ``timeout=True``  -> block indefinitely (Ctrl+C still works).
        * ``timeout=<float>`` -> block up to N seconds; ``None`` on timeout.

        Implementation note: even when ``timeout=True`` we poll the queue
        with a finite 0.5 s timeout in a loop instead of calling
        ``Queue.get(block=True, timeout=None)`` directly.  The latter is
        a known uninterruptible blocking primitive on macOS (and inside
        the Mujoco-bundled ``mjpython`` runtime) where SIGINT cannot be
        delivered to the main thread until the lock returns.
        """
        if timeout is None:
            try:
                return self._queue.get(block=False)
            except Empty:
                return None
        if isinstance(timeout, bool):
            if not timeout:
                try:
                    return self._queue.get(block=False)
                except Empty:
                    return None
            # Indefinite-block path with periodic wake-ups for SIGINT.
            while self._running:
                try:
                    return self._queue.get(block=True, timeout=0.5)
                except Empty:
                    continue
            return None
        try:
            return self._queue.get(block=True, timeout=float(timeout))
        except Empty:
            return None

    def stop(self) -> None:
        self._running = False
        for sock in (self._conn_sock, self._server_sock):
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._conn_sock = None
        self._server_sock = None

    # ------------------------------------------------------------------
    # Network worker
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        try:
            if self.protocol == "tcp":
                self._tcp_run()
            else:
                self._udp_run()
        except Exception as exc:  # pragma: no cover - background thread
            logging.exception(f"[XsensClient] worker exited: {exc}")

    def _tcp_run(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)
        self._server_sock.settimeout(self.accept_timeout)
        if self.verbose:
            print(
                f"[XsensClient] TCP listening on {self.host}:{self.port}, "
                f"waiting for MVN connection..."
            )
        self._conn_sock, addr = self._server_sock.accept()
        self._conn_sock.settimeout(2.0)
        if self.verbose:
            print(f"[XsensClient] Connected from {addr}")

        buf = bytearray()
        last_fps_t = time.time()
        frames_since = 0

        while self._running:
            try:
                data = self._conn_sock.recv(65536)
                if not data:
                    if self.verbose:
                        print("[XsensClient] Connection closed by peer.")
                    break
                buf.extend(data)
            except socket.timeout:
                continue
            except (OSError, BrokenPipeError) as exc:
                if self.verbose:
                    print(f"[XsensClient] socket error: {exc}")
                break

            # Drain as many complete packets as the buffer currently holds.
            while True:
                consumed = self._try_parse_one(buf)
                if not consumed:
                    break
                frames_since += 1

            now = time.time()
            if now - last_fps_t >= 1.0 and self.verbose:
                self.fps = frames_since / (now - last_fps_t)
                print(f"[XsensClient] FPS: {self.fps:.2f}  queued: {self._queue.qsize()}")
                last_fps_t = now
                frames_since = 0

    def _udp_run(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.settimeout(2.0)
        if self.verbose:
            print(f"[XsensClient] UDP listening on {self.host}:{self.port}")

        last_fps_t = time.time()
        frames_since = 0
        while self._running:
            try:
                data, _ = self._server_sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError as exc:
                if self.verbose:
                    print(f"[XsensClient] socket error: {exc}")
                break
            buf = bytearray(data)
            while self._try_parse_one(buf):
                frames_since += 1
            now = time.time()
            if now - last_fps_t >= 1.0 and self.verbose:
                self.fps = frames_since / (now - last_fps_t)
                print(f"[XsensClient] FPS: {self.fps:.2f}  queued: {self._queue.qsize()}")
                last_fps_t = now
                frames_since = 0

    # ------------------------------------------------------------------
    # Packet framing & parsing
    # ------------------------------------------------------------------
    def _try_parse_one(self, buf: bytearray) -> bool:
        """Try to consume one complete MXTPxx packet from ``buf``.

        Returns ``True`` if a packet was consumed (regardless of whether
        we handled it), ``False`` if we need more bytes.
        """
        # Need at least the header
        if len(buf) < _HEADER_SIZE:
            return False

        # Re-sync to MXTP magic if necessary
        if bytes(buf[:4]) != _PACKET_MAGIC:
            idx = bytes(buf).find(_PACKET_MAGIC)
            if idx < 0:
                # No magic in buffer -> keep last 3 bytes (potential split)
                if len(buf) > 3:
                    del buf[:-3]
                return False
            del buf[:idx]
            if len(buf) < _HEADER_SIZE:
                return False

        # Parse the header
        try:
            message_id = bytes(buf[:6]).decode("ascii")
            message_type = int(message_id[-2:])
        except (UnicodeDecodeError, ValueError):
            # Header looked like MXTP but message-type wasn't numeric -> drop 1 byte
            del buf[:1]
            self._dropped_partial_packets += 1
            return True  # we did consume a byte; loop again

        num_items = buf[11]
        try:
            payload_size = struct.unpack(">H", bytes(buf[22:24]))[0]
        except struct.error:
            return False

        # Some MVN versions / setups may leave payload_size==0; reconstruct
        # the size from num_items for the pose-quaternion type at least.
        if message_type == 2:
            est_payload = num_items * _SEGMENT_SIZE
            packet_size = _HEADER_SIZE + (
                payload_size if payload_size > 0 else est_payload
            )
        else:
            packet_size = _HEADER_SIZE + max(payload_size, 0)

        if packet_size <= _HEADER_SIZE or packet_size > 1 << 20:
            # Sanity: implausible size -> drop one byte and re-sync.
            del buf[:1]
            self._dropped_partial_packets += 1
            return True

        if len(buf) < packet_size:
            return False  # need more bytes

        packet = bytes(buf[:packet_size])
        del buf[:packet_size]

        if message_type == 2:
            self._handle_pose_quaternion(packet, num_items)
        # Other message types (01 Euler, 20 joint angles, 25 time code, ...)
        # are ignored on purpose.
        return True

    def _handle_pose_quaternion(self, packet: bytes, num_items: int) -> None:
        frame: dict = {}
        # Cap loop in case num_items lies about the payload.
        max_segments = min(num_items, (len(packet) - _HEADER_SIZE) // _SEGMENT_SIZE)
        for i in range(max_segments):
            base = _HEADER_SIZE + i * _SEGMENT_SIZE
            try:
                seg_id = struct.unpack_from(">I", packet, base)[0]
                (px, py, pz, qw, qx, qy, qz) = struct.unpack_from(
                    ">fffffff", packet, base + 4
                )
            except struct.error:
                return

            idx = int(seg_id) - 1  # spec: ID is 1-based
            body_name = XSENS_SEGMENT_TO_BODY.get(idx)
            if body_name is None:
                continue

            quat = np.array([qw, qx, qy, qz], dtype=np.float64)
            n = float(np.linalg.norm(quat))
            if n < 1e-8 or not np.isfinite(n):
                continue
            quat /= n
            pos = np.array([px, py, pz], dtype=np.float64)
            if not np.all(np.isfinite(pos)):
                continue
            frame[body_name] = (pos, quat)

        if self.require_all_bodies and not _REQUIRED_BODIES.issubset(frame.keys()):
            # Incomplete frame - skip (likely fingers-only / props packet
            # or sensor still warming up).
            return
        if "Hips" not in frame:
            return

        self._frame_count += 1
        self._push_frame(frame)

    def _push_frame(self, frame: dict) -> None:
        """Push to queue, dropping the oldest entry if it is full so that
        the consumer always sees the most recent frame."""
        try:
            self._queue.put(frame, block=False)
        except Exception:
            try:
                self._queue.get(block=False)
            except Empty:
                pass
            try:
                self._queue.put(frame, block=False)
            except Exception:
                pass


# ----------------------------------------------------------------------
# Manual smoke test:  python -m deploy.xsens.client --port 9763
# ----------------------------------------------------------------------
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9763)
    parser.add_argument("--protocol", default="tcp", choices=["tcp", "udp"])
    parser.add_argument("--print-every", type=float, default=1.0,
                        help="seconds between debug prints (default 1.0)")
    args = parser.parse_args()

    client = XsensClient(host=args.host, port=args.port, protocol=args.protocol)
    client.start_thread()
    last = time.time()
    try:
        while True:
            frame = client.get_frame_data(timeout=0.5)
            if frame is None:
                continue
            now = time.time()
            if now - last >= args.print_every:
                last = now
                hips_pos, hips_quat = frame["Hips"]
                lh_pos, _ = frame.get("LeftHand", (np.zeros(3), np.zeros(4)))
                rh_pos, _ = frame.get("RightHand", (np.zeros(3), np.zeros(4)))
                print(
                    f"frame#{client._frame_count}  "
                    f"hips_pos={hips_pos.round(3).tolist()}  "
                    f"hips_quat={hips_quat.round(3).tolist()}  "
                    f"LH={lh_pos.round(3).tolist()}  "
                    f"RH={rh_pos.round(3).tolist()}  "
                    f"bodies={len(frame)}  fps={client.fps:.1f}"
                )
    except KeyboardInterrupt:
        print()
    finally:
        client.stop()


if __name__ == "__main__":
    _main()
