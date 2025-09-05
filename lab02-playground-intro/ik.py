import mujoco
import numpy as np


class IKSolver:
    '''
    Very simple inverse kinematics (IK) solver for the Unitree Go2 robot using its front right leg. 
        To learn more about this type of IK, see: https://manipulation.csail.mit.edu/trajectories.html#section1 or
        https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    '''
    def __init__(self):
        self.model_cpu = mujoco.MjModel.from_xml_path("unitree_go2/scene_mjx.xml")
        self.data = mujoco.MjData(self.model_cpu)
        self.nv = self.model_cpu.nv
        
        self.foot_site_id = mujoco.mj_name2id(self.model_cpu, mujoco.mjtObj.mjOBJ_SITE, "FR_foot")
        if self.foot_site_id == -1:
            raise ValueError("RL_foot site not found")
        
        self.joint_names = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"]


        self.dof_idxs = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model_cpu, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint {name} not found")
            dofdofadr = self.model_cpu.jnt_dofadr[jid]  # index into qvel

            self.dof_idxs.append(dofdofadr)
    
    def forward_foot(self):
        return self.data.site_xpos[self.foot_site_id]        

    def compute_jacobian(self):
        mujoco.mj_forward(self.model_cpu, self.data)
        
        jacp = np.zeros((3, self.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model_cpu, self.data, jacp, None, self.foot_site_id)
        
        Jp_sub = jacp[:, self.dof_idxs]   # Translational 3×3

        return Jp_sub
    
    def solve_ik(self, init_q, pos_des, lr=0.2, damping=1e-2, tol=1e-4, max_iters=50):
        q = np.array(init_q, copy=True) # (19, )
        self.data.qpos[:] = q
        
        for i in range(max_iters):
            mujoco.mj_forward(self.model_cpu, self.data)
            
            p_cur = self.forward_foot()
            err = pos_des - p_cur
            
            if np.linalg.norm(err) < tol:
                break
            
            J = self.compute_jacobian()  # 3×3
            H = J @ J.T + damping * np.eye(3)
            lam = np.linalg.solve(H, err)
            dq = J.T @ lam
            
            
            q[np.array(self.dof_idxs)+1] += lr * dq
            self.data.qpos[:] = q
            
        return q