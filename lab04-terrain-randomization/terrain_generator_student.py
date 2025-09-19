import xml.etree.ElementTree as xml_et
import numpy as np
import cv2
import noise
import os

def euler_to_quat(roll, pitch, yaw):
    """
    Convert ZYX Euler angles to a quaternion [w, x, y, z].

    @param roll  rotation around x-axis [rad]
    @param pitch rotation around y-axis [rad]
    @param yaw   rotation around z-axis [rad]
    @return np.ndarray (4,) quaternion [w, x, y, z]
    """
    
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def euler_to_rot(roll, pitch, yaw):
    """
    Convert ZYX Euler angles to a 3x3 rotation matrix.

    @param roll  rotation around x-axis [rad]
    @param pitch rotation around y-axis [rad]
    @param yaw   rotation around z-axis [rad]
    @return np.ndarray (3, 3) rotation matrix
    """
    
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x



def rot2d(x, y, yaw):
    """
    Rotate a 2D point (x, y) by angle yaw.

    @param x    x-coordinate
    @param y    y-coordinate
    @param yaw  rotation angle [rad]
    @return (float, float) rotated (x, y)
    """
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny



def rot3d(pos, euler):
    """
    Rotate a 3D vector by ZYX Euler angles.

    @param pos   np.ndarray (3,) vector
    @param euler np.ndarray (3,) [roll, pitch, yaw]
    @return np.ndarray (3,) rotated vector
    """
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


def list_to_str(vec):
    return " ".join(str(s) for s in vec)


class TerrainGenerator:

    def __init__(self, input_scene_path, output_scene_path) -> None:
        self.input_scene_path=input_scene_path
        self.output_scene_path = output_scene_path
        
        self.scene = xml_et.parse(input_scene_path)
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")

    def AddBox(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1, 0.1]):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
    
    def AddGeometry(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1],
               geo_type="box" # "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
               ):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddStairs(self,
                  init_pos=[1.0, 0.0, 0.0],
                  yaw=0.0,
                  width=0.2,
                  height=0.15,
                  length=1.5,
                  stair_nums=10):
        '''
        TODO(student): Impliment this function
        
        @param init_pos     [start_x, start_y, start_z] position of the first stair
        @param yaw          rotation of stairs around z axis [rad]
        @param width        width of each stair along the x axis
        @param height       height of each stair along the z axis
        @param length       length of each stair along the y axis
        @param stair_nums   number of stairs
        '''
        
        for i in range(stair_nums):
            # find per-stair position
            stair_x = init_pos[0] + i * width
            stair_y = init_pos[1]
            stair_z = init_pos[2] + (i + 1) * height
            
            if yaw != 0.0:
                # relative to init_pos
                rel_x = stair_x - init_pos[0]
                rel_y = stair_y - init_pos[1]
                rotated_x, rotated_y = rot2d(rel_x, rel_y, yaw)
                stair_x = init_pos[0] + rotated_x
                stair_y = init_pos[1] + rotated_y
            
            position = [stair_x, stair_y, stair_z]
            euler = [0.0, 0.0, yaw]
            size = [width, length, (i + 1) * height * 2] 
            
            # create stair
            self.AddBox(position=position, euler=euler, size=size)

    def AddRoughGround(self,
                       init_pos=[1.0, 0.0, 0.0],
                       euler=[0.0, -0.0, 0.0],
                       nums=[10, 10],
                       box_size=[0.5, 0.5, 0.5],
                       box_euler=[0.0, 0.0, 0.0],
                       separation=[0.2, 0.2],
                       box_size_rand=[0.05, 0.05, 0.05],
                       box_euler_rand=[0.2, 0.2, 0.2],
                       separation_rand=[0.05, 0.05]):
        
        '''
        TODO(student): Impliment this function
        
        @param init_pos         [start_x, start_y, start_z] position of the first box
        @param euler            rotation of the whole rough ground around x, y, z axis
        @param nums             [num_x, num_y] number of boxes along x and y axis
        @param box_size         [size_x, size_y, size_z] size of each box
        @param box_euler        [roll, pitch, yaw] rotation of each box around x, y, z axis
        @param separation       [sep_x, sep_y] separation between boxes along x and y axis
        @param box_size_rand    [rand_x, rand_y, rand_z] random range of box size
        @param box_euler_rand   [rand_roll, rand_pitch, rand_yaw] random range of box euler
        @param separation_rand  [rand_sep_x, rand_sep_y] random range of separation
        '''
        
        for i in range(nums[0]):  # Loop over x
            for j in range(nums[1]):  # Loop over y

                # base box pos
                x_init = i * (box_size[0] + separation[0])
                y_init = j * (box_size[1] + separation[1])
                z_init = 0.0
                
                # randomize separation range
                rand_sep_x = np.random.uniform(-separation_rand[0], separation_rand[0])
                rand_sep_y = np.random.uniform(-separation_rand[1], separation_rand[1])
                
                local_pos = [x_init + rand_sep_x, y_init + rand_sep_y, z_init]
                
                # ror local position by the overall euler angles
                if any(e != 0.0 for e in euler):
                    local_pos = rot3d(np.array(local_pos), np.array(euler))
                
                # calculate final pos
                final_pos = [
                    init_pos[0] + local_pos[0],
                    init_pos[1] + local_pos[1], 
                    init_pos[2] + local_pos[2] + box_size[2]/2  # Offset by half height so box sits on ground
                ]
                
                # randomize size
                rand_size = [
                    box_size[0] + np.random.uniform(-box_size_rand[0], box_size_rand[0]),
                    box_size[1] + np.random.uniform(-box_size_rand[1], box_size_rand[1]),
                    box_size[2] + np.random.uniform(-box_size_rand[2], box_size_rand[2])
                ]
                
                # randomize rotation
                rand_euler = [
                    box_euler[0] + np.random.uniform(-box_euler_rand[0], box_euler_rand[0]),
                    box_euler[1] + np.random.uniform(-box_euler_rand[1], box_euler_rand[1]),
                    box_euler[2] + np.random.uniform(-box_euler_rand[2], box_euler_rand[2])
                ]
                
                # get net rotation too
                final_euler = [
                    euler[0] + rand_euler[0],
                    euler[1] + rand_euler[1], 
                    euler[2] + rand_euler[2]
                ]
                
                # generate box subterrain element
                self.AddBox(position=final_pos, euler=final_euler, size=rand_size)

    def AddPerlinHeighField(
            self,
            position=[1.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[1.0, 1.0],  # width and length
            height_scale=0.2,  # max height
            negative_height=0.2,  # height in the negative direction of z axis
            image_width=128,  # height field image size
            img_height=128,
            smooth=100.0,  # smooth scale
            perlin_octaves=6,  # perlin noise parameter
            perlin_persistence=0.5,
            perlin_lacunarity=2.0,
            output_hfield_image="height_field.png"):
        
        '''
        TODO(student): Impliment this function
        
        @param position             [x, y, z] position of the height field
        @param euler                [roll, pitch, yaw] rotation of the height field around x, y, z axis
        @param size                 [size_x, size_y] size of the height field along x and y axis
        @param height_scale         max height of the height field
        @param negative_height      height in the negative direction of z axis
        @param image_width          width of the generated height field image
        @param img_height           height of the generated height field image
        @param smooth               smooth scale of the perlin noise
        @param perlin_octaves       perlin noise parameter
        @param perlin_persistence   perlin noise parameter
        @param perlin_lacunarity    perlin noise parameter
        @param output_hfield_image  output height field image file name
        
        HINT: Look at the noise implimentation: https://github.com/caseman/noise
              you can also run "help(noise)" to look at the function signatures. 
              You'll likely want to use noise.pnoise2(). 
              
              Instead of just using the x, y, coordinates as inputs to noise.pnoise2() 
              directly, you should scale them down by the "smooth" parameter
        '''
        
        print("Not Implimented: AddPerlinHeighField")
        
        

        # TODO(student): Generate height field based on perlin noise
        terrain_image = None
        
        
        hfield_save_file = os.path.join(os.path.dirname(self.output_scene_path), output_hfield_image)
        cv2.imwrite(hfield_save_file, terrain_image)

        # Create the height field asset 
        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "./" + output_hfield_image


        #TODO(student): Make a new heightfield geom as a sub-element of self.worldbody. 
        # It should have the following attributes: (type, hfield, pos, quat). The type attribute should be "hfield" and
        # the hfield attribute should be the name of the height field asset you just created above.

        
    def AddHeighFieldFromImage(
            self,
            position=[1.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[2.0, 1.6],  # width and length
            height_scale=0.02,  # max height
            negative_height=0.1,  # height in the negative direction of z axis
            input_img=None,
            output_hfield_image="height_field.png",
            image_scale=[1.0, 1.0],  # reduce image resolution
            invert_gray=False):
        
        print("Not Implimented: AddHeighFieldFromImage")
        

        
    def CustomTerrain(self):
        '''
        TODO(student): feel free to impliment your own custom modular terrain component here!
        '''
        
        print("Not Implimented: CustomTerrain")

    def Save(self):
        self.scene.write(self.output_scene_path)


if __name__ == "__main__":
    input_scene_path = "./google_barkour_vb/scene_mjx.xml"
    output_scene_path = "./google_barkour_vb/scene_mjx_with_terrain.xml"
    tg = TerrainGenerator(input_scene_path, output_scene_path)

    # Box obstacle
    tg.AddBox(position=[1.5, 0.0, 0.1], euler=[0, 0, 0.0], size=[1, 1.5, 0.2])
    
    # Geometry obstacle
    # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
    tg.AddGeometry(position=[1.5, 0.0, 0.25], euler=[0, 0, 0.0], size=[1.0,0.5,0.5],geo_type="cylinder")

    # Slope
    tg.AddBox(position=[2.0, 2.0, 0.5],
              euler=[0.0, -0.5, 0.0],
              size=[3, 1.5, 0.1])

    # Stairs
    tg.AddStairs(init_pos=[1.0, 4.0, 0.0], yaw=0.0)

    # Rough ground
    tg.AddRoughGround(init_pos=[-2.5, 5.0, 0.0],
                      euler=[0, 0, 0.0],
                      nums=[10, 8])

    # Perlin heigh field
    tg.AddPerlinHeighField(position=[-1.5, 4.0, 0.0], size=[2.0, 1.5])

    tg.Save()
