import argparse
import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
from mujoco import mjx

from brax import envs
from custom_env import UnitreeGo2Env    # from custom_env.py
from ik import IKSolver                 # from ik.py

envs.register_environment('unitreego2', UnitreeGo2Env)

def intro_to_mujoco_and_mjx():
    """ Corresponds to 'Introduction to Mujoco & MJX.'
        Contains the TODO about the green sphere
    """

    # Define Mujoco xml
    xml = """
    <mujoco>
      <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
          <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
          <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
          <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    # Init Mujoco model and MjData state data
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model)

    # Init some stuff for both CPU and GPU env stepping
    duration = 5    # (s)
    framerate = 60  # (Hz)
    frames = []
    mujoco.mj_resetData(mj_model, mj_data)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True


    # Use pure MuJoCo (CPU) to step thru env
    while mj_data.time < duration:
        mujoco.mj_step(mj_model, mj_data)
        if len(frames) < mj_data.time * framerate:
            renderer.update_scene(mj_data)
            pixels = renderer.render()
            frames.append(pixels)

    # Save video
    print("Saving video to intro_simulation.mp4")
    media.write_video('intro_simulation.mp4', frames, fps=framerate)
    
    # Render the simple environment rollout on GPU
    print("Attempting Dockerized local GPU inference. Prepare for issues...")
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)
    
    print("Vectorized data:")
    print("mj_data.qpos: ", mj_data.qpos, type(mj_data.qpos))
    print("mjx_data.qpos:", mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

    # This time use MJX to step thru env
    jit_step = jax.jit(mjx.step)
    frames = []
    mujoco.mj_resetData(mj_model, mj_data)
    mjx_data = mjx.put_data(mj_model, mj_data)
    while mjx_data.time < duration:

        mjx_data = jit_step(mjx_model, mjx_data)
        if len(frames) < mjx_data.time * framerate:
            mj_data = mjx.get_data(mj_model, mjx_data)
            renderer.update_scene(mj_data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    # Save video
    print("Saving video to intro_simulation_gpu.mp4")
    media.write_video('intro_simulation_gpu.mp4', frames, fps=framerate)

    # Close render
    renderer.close()

    """
    TASK 1: Describe in your own words how jax random numbers work. What is the purpose of `jax.random.split`? 
    (5 pts)

    ANSWER: 
    Random numbers in JAX, like all random numbers in code, are really just pseudorandom numbers.
    JAX specifically uses PRNGKey to generate random numbers.
            
    `jax.random.split` is JAX's way of generating new random number seeds using an old one.
    Due to the pseudorandom nature of random num. generation, you can never reuse a seed.
    
    Therefore, you call 'jax.random.split' to turn that one seed into multiple subkeys, 
    use the first subkey, save the rest for later, and now your pseudo-random-ness is kept
    clean, as close to random as possible, and 100% reproducible.
    """
    # ======================================================================
    # TODO (student): given the name of a geom, access its color
    # use the function mujoco.mj_name2id and the mj_model.geom_rgba property
    geom_name = "green_sphere"
    
    # Get unique ID using mj_name2id
    geom_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_GEOM, # Specify to look for a geom
        geom_name
    )

    # Get color using ID to index
    geom_color = mj_model.geom_rgba[geom_id]

    answer = geom_color
    print("Answer = ", answer) # [0. 1. 0. 1.] AKA full green, alpha=1
    # ======================================================================

def loading_an_environment_and_agent():
    """ Corresponds to 'Loading an Environment and Agent'
    """
    # instantiate the env
    env_name = 'unitreego2'
    env = envs.get_environment(env_name)

    # jit reset/step functions for fast runtime
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # reset the sim env using seed zero
    state = jit_reset(jax.random.PRNGKey(0))
    plt.imshow(env.render([state.pipeline_state], camera='track')[0])

    """ 
    TASK 2: 
    Similarly, we can rollout an entire trajectory in our environment using the `jit_step()` function. 
    (10 pts) 
    """

    state = jit_reset(jax.random.PRNGKey(0))

    rollout = [state.pipeline_state]

    # grab a trajectory
    for i in range(500):
        
        # TODO(student): Sample the ZERO action for the robot. For now, just create a `jax.array` 
        # with the same size as the number of actuators. Set each element of the array to be 0. 
        # hint 1: you can use the `env.sys.nu` property to get the number of actuators/motors
        # hint 2: the `jp` module is an alias for jax.numpy which has similar functionality to 
        # numpy (i.e. jp.zeros, jp.ones, jp.array, etc)
        action = jp.zeros(12) # Unitree Go2 is 12-DOF
        
        # TODO(student): Take a "jit_step" in the environment. See the custom_env.py file for details 
        # on the arguments the function takes and what it returns. 
        state = jit_step(state=state, action=action) # Literally just our zero action over and over
        
        rollout.append(state.pipeline_state)

    media.show_video(env.render(rollout, camera="track"), fps=1.0 / env.dt)

if __name__ == "__main__":
    print(
        "\n"
        "====================== NOTE TO GRADER ======================\n"
        "============================================================\n"
        "Next time, I promise I'll make the Jupyter Notebooks\n" \
        "run on their own (to make grading easier).\n"
        "But for now, please enjoy  b i g   m a i n   m e t h o d  👍\n"
        "============================================================\n"
        "============================================================\n"
    )

    intro_to_mujoco_and_mjx()

