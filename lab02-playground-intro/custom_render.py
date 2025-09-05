'''Custom mujoco renderer that draws foot trace for the Unitree Go2 robot.'''

import numpy as np
import mujoco

from typing import Optional, Union, List, Sequence
import brax
from brax import base

def render_array_with_foot_trace(
    sys: brax.System,
    trajectory: Union[List[base.State], base.State],
    height: int = 240,
    width: int = 320,
    camera: Optional[str] = None,
    render_until_timestep = 10000
) -> Union[Sequence[np.ndarray], np.ndarray]:
    """Returns a sequence of np.ndarray images using the MuJoCo renderer."""
    renderer = mujoco.Renderer(sys.mj_model, height=height, width=width)
    camera = camera or -1
  
    positions = []

    frames = []
    for state in trajectory:
        d = mujoco.MjData(sys.mj_model)
        d.qpos, d.qvel = state.q, state.qd
        if hasattr(state, 'mocap_pos') and hasattr(state, 'mocap_quat'):
            d.mocap_pos, d.mocap_quat = state.mocap_pos, state.mocap_quat
            mujoco.mj_forward(sys.mj_model, d)
            
            fr_foot_pos = d.geom_xpos[d.geom("FR").id].copy()
            positions.append(fr_foot_pos)
            
            renderer.update_scene(d, camera=camera)
            modify_scene(renderer.scene, positions, render_until_timestep)
            
            frames.append(renderer.render())
        
    return frames

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_connector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                        point1, point2)
  
def modify_scene(scn, positions, render_until_timestep):
    """Draw position trace in the render scene"""
    
    if len(positions) > 1:
        for i in range(len(positions)-1):
            rgba=np.array((1.0, 0.0, 0.0, 1.0))
            radius=.007*(1)
            point1 = positions[i] #+ offset*times[i]
            point2 = positions[i+1] #+ offset*times[i+1]
            if i < render_until_timestep:
                add_visual_capsule(scn, point1, point2, radius, rgba)
            
            
            
