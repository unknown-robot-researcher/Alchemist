import rospy
from my_robot_package.FunctionLibrary import FunctionLib

# Initialize the ROS node
rospy.init_node("gpt")

# Initialize the function library
lib = FunctionLib()

# Move the robot to the home position first
lib.move_to_home_position()

# Get the current end-effector pose
current_pose = lib.get_current_end_effector_pose()
x, y, z, roll, pitch, yaw = current_pose

# Move the end-effector up by 5 cm
new_z = z + 0.05
lib.go(x, y, new_z, roll, pitch, yaw)

# Move back to the home position
lib.move_to_home_position()

# Print hurray
print("Hurray")
