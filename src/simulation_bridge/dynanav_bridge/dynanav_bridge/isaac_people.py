"""Isaac Sim 5.1 compatibility wrapper for DynaNav dynamic characters."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import yaml


PEOPLE_EXTENSIONS = (
    "omni.anim.timeline",
    "omni.anim.graph.core",
    "omni.anim.retarget.core",
    "omni.anim.navigation.core",
    "omni.anim.people",
    "isaacsim.replicator.agent.core",
)

NAVMESH_VOLUMES = {
    "hospital": ((0.0, 0.0, 5.0), (120.0, 120.0, 20.0)),
    "office": ((0.0, 0.0, 3.0), (50.0, 50.0, 12.0)),
    "outdoor": ((0.0, 100.0, 10.0), (500.0, 500.0, 40.0)),
    "warehouse": ((0.0, 0.0, 5.0), (120.0, 120.0, 20.0)),
}


class DynaNavPeopleRuntime:
    """Spawn and animate DynaNav characters without replacing the active IsaacLab stage."""

    def __init__(self, workspace_root: str, scene: str, scene_usd: str, count: int, seed: int):
        self.workspace_root = Path(workspace_root)
        self.scene = str(scene)
        self.scene_usd = str(scene_usd)
        self.count = int(count)
        self.seed = int(seed)
        self.try_isaac_people = os.environ.get("ROBOT_VLN_TRY_ISAAC_PEOPLE", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self._manager = None
        self._position_manager = None
        self._setup_subscription = None
        self._kinematic_people = {}
        self._initial_positions = {}
        self._commands_text = ""
        self.mode = "disabled"
        self._tempdir = tempfile.TemporaryDirectory(prefix="robot_vln_dynanav_people_")

    def enable_extensions(self, simulation_app) -> None:
        if not self.try_isaac_people:
            return
        import carb
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        for extension in PEOPLE_EXTENSIONS:
            extension_manager.set_extension_enabled_immediate(extension, True)
        settings = carb.settings.get_settings()
        settings.set("/persistent/scripting/enablePythonScripting", True)
        settings.set("/persistent/scripting/allowPythonScripting", True)
        settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
        settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentHeight", 180)
        settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentRadius", 40)
        settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxStepHeight", 20)
        settings.set("/exts/omni.anim.navigation.core/navMesh/config/agentMaxFloorSlope", 45.0)
        for _ in range(4):
            simulation_app.update()

    def _write_inputs(self) -> str:
        dynanav_root = self.workspace_root / "third_party" / "TIC-VLA" / "DynaNav"
        if str(dynanav_root) not in sys.path:
            sys.path.insert(0, str(dynanav_root))
        from generate_commands import generate_character_commands

        temp_root = Path(self._tempdir.name)
        command_path = temp_root / "character_commands.txt"
        self._commands_text = generate_character_commands(
            self.scene,
            num_characters=self.count,
            num_commands=10,
            seed=self.seed,
            filename=str(command_path),
            base_name="Character",
        )
        config = {
            "isaacsim.replicator.agent": {
                "version": "0.7.28",
                "global": {"seed": self.seed, "simulation_length": 86400},
                "scene": {"asset_path": self.scene_usd},
                "sensor": {"camera_num": 0},
                "character": {"command_file": str(command_path), "num": self.count},
                "replicator": {
                    "writer": "IRABasicWriter",
                    "parameters": {
                        "output_dir": str(temp_root / "unused_output"),
                        "rgb": False,
                        "camera_params": False,
                    },
                },
            }
        }
        config_path = temp_root / "people_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return str(config_path)

    def _setup_kinematic_people(self) -> None:
        """Fallback for referenced stages whose NavMesh cannot be baked in IsaacLab."""
        import omni.usd
        from pxr import Gf, UsdGeom

        from generate_commands import TARGET_SETS

        targets = list(TARGET_SETS[self.scene].values())
        commands = {}
        for line in self._commands_text.splitlines():
            fields = line.split()
            if len(fields) >= 5 and fields[1] == "GoTo":
                commands.setdefault(fields[0], []).append(tuple(float(value) for value in fields[2:5]))
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/Characters")
        rng = random.Random(self.seed)
        offset = rng.randrange(len(targets))
        for index in range(self.count):
            name = "Character" if index == 0 else f"Character_{index:02d}"
            path = f"/World/Characters/{name}"
            route = commands.get(name, [])
            start = tuple(float(value) for value in targets[(offset + index) % len(targets)])
            if not route:
                route = [tuple(float(value) for value in targets[(offset + index + 1) % len(targets)])]
            root = UsdGeom.Xform.Define(stage, path)
            translate = root.AddTranslateOp()
            translate.Set(Gf.Vec3d(*start))
            body = UsdGeom.Capsule.Define(stage, f"{path}/Body")
            body.CreateAxisAttr("Z")
            body.CreateHeightAttr(1.15)
            body.CreateRadiusAttr(0.24)
            body.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.82))
            body.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.42, 0.78)])
            head = UsdGeom.Sphere.Define(stage, f"{path}/Head")
            head.CreateRadiusAttr(0.19)
            head.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 1.62))
            head.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.58, 0.42)])
            self._kinematic_people[path] = {
                "position": list(start),
                "route": route,
                "route_index": 0,
                "translate": translate,
            }
            self._initial_positions[path] = start
        self.mode = "kinematic_compatibility"
        print(
            f"[DYNANAV PEOPLE] READY scene={self.scene} count={self.count} seed={self.seed} "
            "mode=kinematic_compatibility",
            flush=True,
        )

    def _ensure_navmesh_volume(self) -> None:
        """Author a session-only volume when the referenced scene has none."""
        import omni.kit.commands
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/NavMeshVolume").IsValid():
            return
        if self.scene not in NAVMESH_VOLUMES:
            raise ValueError(f"No NavMeshVolume bounds configured for scene {self.scene!r}")
        center, scale = NAVMESH_VOLUMES[self.scene]
        success, _ = omni.kit.commands.execute(
            "CreateNavMeshVolumeCommand",
            parent_prim_path=Sdf.Path("/World"),
            volume_type=0,
            position=Gf.Vec3d(*center),
        )
        if not success:
            raise RuntimeError("Isaac Sim failed to create the DynaNav NavMeshVolume")
        prim = stage.GetPrimAtPath("/NavMeshVolume")
        if not prim.IsValid():
            raise RuntimeError("CreateNavMeshVolumeCommand did not create /NavMeshVolume")
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))
        print(
            f"[DYNANAV PEOPLE] Added session NavMeshVolume center={center} scale={scale}",
            flush=True,
        )

    def setup(self, simulation_app, timeout: float = 300.0) -> None:
        if self.count <= 0:
            return
        if not self.try_isaac_people:
            self._write_inputs()
            print(
                "[DYNANAV PEOPLE] Isaac People/NavMesh is disabled by default on Isaac Sim 5.1; "
                "using the deterministic compatibility runtime. Set ROBOT_VLN_TRY_ISAAC_PEOPLE=1 to retest it.",
                flush=True,
            )
            self._setup_kinematic_people()
            return
        from isaacsim.replicator.agent.core.simulation import SimulationManager
        from omni.anim.people.scripts.global_character_position_manager import GlobalCharacterPositionManager
        import omni.usd

        config_path = self._write_inputs()
        self._ensure_navmesh_volume()
        for _ in range(3):
            simulation_app.update()
        import omni.anim.navigation.core as nav

        navigation = nav.acquire_interface()
        navigation.start_navmesh_baking_and_wait()
        if navigation.get_navmesh() is None:
            self._setup_kinematic_people()
            return
        self._manager = SimulationManager()
        if not self._manager.load_config_file(config_path):
            raise RuntimeError(f"DynaNav people config could not be loaded: {config_path}")

        setup_done = False

        def on_setup_done(_event):
            nonlocal setup_done
            setup_done = True

        self._setup_subscription = self._manager.register_set_up_simulation_done_callback(on_setup_done)
        self._manager.load_assets_to_scene()
        deadline = time.monotonic() + float(timeout)
        while not setup_done and simulation_app.is_running() and time.monotonic() < deadline:
            simulation_app.update()
        self._setup_subscription = None
        if not setup_done:
            raise RuntimeError(
                "DynaNav people setup timed out; the scene may lack a valid NavMeshVolume or online people assets"
            )

        stage = omni.usd.get_context().get_stage()
        character_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetName().startswith("Character_") and prim.GetTypeName() == "Xform"
        ]
        if len(character_paths) < self.count:
            raise RuntimeError(f"Expected {self.count} DynaNav characters, found {len(character_paths)}")
        self._position_manager = GlobalCharacterPositionManager.get_instance()
        self.mode = "isaac_people"
        print(
            f"[DYNANAV PEOPLE] READY scene={self.scene} count={self.count} seed={self.seed} mode=isaac_people",
            flush=True,
        )

    def advance(self, dt: float) -> None:
        if not self._kinematic_people:
            return
        from pxr import Gf

        step = 0.8 * float(dt)
        for state in self._kinematic_people.values():
            route = state["route"]
            target = route[state["route_index"]]
            position = state["position"]
            dx, dy = target[0] - position[0], target[1] - position[1]
            distance = math.hypot(dx, dy)
            if distance <= max(step, 0.05):
                position[:] = [target[0], target[1], target[2]]
                state["route_index"] = (state["route_index"] + 1) % len(route)
            elif distance > 0.0:
                position[0] += step * dx / distance
                position[1] += step * dy / distance
                position[2] = target[2]
            state["translate"].Set(Gf.Vec3d(*position))

    def contacts(self, robot_position, threshold: float = 0.65):
        """Return distance-based pedestrian contacts using omni.anim.people state."""
        robot_x, robot_y = float(robot_position[0]), float(robot_position[1])
        contacts = []
        if self._position_manager is not None:
            positions = {
                str(path): self._position_manager.get_character_current_pos(path)
                for path in self._position_manager.get_all_managed_characters()
            }
        else:
            positions = {path: state["position"] for path, state in self._kinematic_people.items()}
        for path, position in positions.items():
            distance = math.hypot(float(position[0]) - robot_x, float(position[1]) - robot_y)
            if distance <= threshold:
                contacts.append(
                    {
                        "robot_body": f"pedestrian:{path}",
                        "force": 0.0,
                        "object": str(path),
                        "is_pedestrian": True,
                        "distance": distance,
                    }
                )
        return contacts

    def status(self) -> dict:
        positions = {path: list(state["position"]) for path, state in self._kinematic_people.items()}
        if self._position_manager is not None:
            for path in self._position_manager.get_all_managed_characters():
                positions[str(path)] = list(self._position_manager.get_character_current_pos(path))
        return {
            "scene": self.scene,
            "count": self.count,
            "seed": self.seed,
            "mode": self.mode,
            "positions": positions,
        }

    def close(self) -> None:
        if self._kinematic_people:
            max_displacement = max(
                math.hypot(
                    state["position"][0] - self._initial_positions[path][0],
                    state["position"][1] - self._initial_positions[path][1],
                )
                for path, state in self._kinematic_people.items()
            )
            print(f"[DYNANAV PEOPLE] max_displacement={max_displacement:.3f}m", flush=True)
        self._setup_subscription = None
        self._manager = None
        self._position_manager = None
        self._kinematic_people.clear()
        self._tempdir.cleanup()
