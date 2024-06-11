from Lib.xarm7.FunctionLibrary import FunctionLib
import rclpy
import time

# Initialize the ROS node
rclpy.init()
node = rclpy.create_node('gpt')

# Initialize the function library
lib = FunctionLib()

# Get the current end effector pose
current_pose = lib.get_current_end_effector_pose()

# Extract the current Z position
current_z = current_pose[2]

# Move the arm up by 0.01 meters
lib.move_arm(current_pose[0], current_pose[1], current_z + 0.01, current_pose[3], current_pose[4], current_pose[5])

# Wait for a short period to ensure the movement completes
time.sleep(1.0)

# Move the arm down by 0.01 meters (back to the original position)
lib.move_arm(current_pose[0], current_pose[1], current_z - 0.01, current_pose[3], current_pose[4], current_pose[5])

# Wait for a short period to ensure the movement completes
time.sleep(1.0)

# Move the arm up again by 0.01 meters
lib.move_arm(current_pose[0], current_pose[1], current_z + 0.01, current_pose[3], current_pose[4], current_pose[5])

# Wait for a short period to ensure the movement completes
time.sleep(1.0)

# Move the arm back to the original Z position
lib.move_arm(current_pose[0], current_pose[1], current_z, current_pose[3], current_pose[4], current_pose[5])

# Print something funny
print("I'm a little teapot, short and stout!")

# Move the arm back to the home position (assuming home position is x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
lib.move_arm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
