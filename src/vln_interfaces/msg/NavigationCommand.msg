std_msgs/Header header

uint8 WAYPOINT=0
uint8 TRAJECTORY=1
uint8 VELOCITY=2
uint8 STOP=3

uint8 command_type
geometry_msgs/Pose2D[] points
geometry_msgs/Twist velocity
float32 dt
float32 horizon
float32 confidence
bool valid
bool goal_reached
