import numpy as np
from agents.components import LLM
from agents.models import Llama3_1
from agents.vectordbs import ChromaDB
from agents.config import LLMConfig
from agents.clients.roboml import HTTPDBClient
from agents.clients.ollama import OllamaClient
from agents.ros import Launcher, Topic
import json
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# Create QoS profile
qos_profile = {
    'reliability': ReliabilityPolicy.BEST_EFFORT,
    'durability': DurabilityPolicy.VOLATILE,
    'history': HistoryPolicy.KEEP_LAST,
}

# Start a Llama3.1 based llm component using ollama client
llama = Llama3_1(name="llama")
llama_client = OllamaClient(llama)

# Initialize a vector DB that will store our routes
# chroma = ChromaDB(name="MainDB")
# chroma_client = HTTPDBClient(db=chroma, port=9000)

px4_request = Topic(name="text0", msg_type="String")
px4_odom = Topic(name="fmu/out/vehicle_odometry", msg_type="VehicleOdometry", qos_profile=qos_profile)
px4_status = Topic(name="fmu/out/vehicle_status", msg_type="VehicleStatus", qos_profile=qos_profile)
px4_response = Topic(name="text1", msg_type="String")

config = LLMConfig(
    enable_rag=False,
    collection_name="px4_commands",
    distance_func="l2",
    chat_history=False,
    n_results=1,
    add_metadata=False,
    temperature=0.2,
)

# initialize the component
px4 = LLM(
    inputs=[px4_request, px4_odom, px4_status],
    outputs=[px4_response],
    model_client=llama_client,
    # db_client=chroma_client,
    trigger=px4_request,
    config=config,
    component_name="px4_request",
)

def get_vehicle_odometry() -> str:
    """Get the current vehicle odometry data including position and orientation.
    
    :returns:  A JSON string with the current vehicle odometry data
    :rtype:    str
    """
    print("========================================")
    print("TOOL INVOKED: get_vehicle_odometry")
    print("========================================")
    odometry_data = px4.latest_vehicle_odom_response
    print(odometry_data)
    # Convert numpy array to regular list for JSON serialization
    if isinstance(odometry_data, np.ndarray):
        pos_x, pos_y, pos_z = odometry_data[0:3]
        qx, qy, qz, qw = odometry_data[3:7]
        
        # Calculate heading from quaternion (for yaw only)
        heading = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        heading_rad = np.radians(heading)
        
        result = {
            "position": {
                "x": round(float(pos_x), 2),
                "y": round(float(pos_y), 2),
                "z": round(float(pos_z), 2),
                "units": "meters"
            },
            "heading": round(float(heading_rad), 4),
            "heading_units": "radians"
        }
        return json.dumps(result)
    return json.dumps({"error": "No odometry data available"})
    #     return result
    # return "Error: No odometry data available"


odometry_function_description = {
    "type": "function",
    "function": {
        "name": "get_vehicle_odometry",
        "description": "Get the current vehicle position and orientation in 3D space. Use this function to determine where the drone is located or which direction it's facing.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
    }
}

def get_vehicle_status() -> str:
    """Get the current vehicle status.
    
    :returns:  A JSON string with the current vehicle odometry data
    :rtype:    str
    """
    print("========================================")
    print("TOOL INVOKED: get_vehicle_status")
    print("========================================")
    status_data = px4.latest_vehicle_status_response
    print(status_data)
    # Convert numpy array to regular list for JSON serialization
    if isinstance(status_data, np.ndarray):
        # Dictionary to map arming state values to human-readable strings
        result = {
            "arming_state": status_data.get('arming', 'unknown'),
            "navigation_state": status_data.get('nav_state', 'unknown'),
            "failure_detector": status_data.get('failure_detector_status', 'none'),
            "failsafe": status_data.get('failsafe', 'unknown')
        }
        return json.dumps(result)
    return json.dumps({"error": "No status data available"})
    #     return result
    # return "Error: No status data available"

status_function_description = {
    "type": "function",
    "function": {
        "name": "get_vehicle_status",
        "description": "Get the current vehicle status including arming state, navigation mode, and system health. Use this function to check if the drone is armed, what mode it's in, or if there are any failures.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
    }
}

px4.register_tool(
    tool=get_vehicle_odometry,
    tool_description=odometry_function_description,
    send_tool_response_to_model=True,
)

px4.register_tool(
    tool=get_vehicle_status,
    tool_description=status_function_description,
    send_tool_response_to_model=True,
)

# px4.set_topic_prompt(
    
# )

# px4.set_component_prompt(template="""

# """)

# px4.set_system_prompt(

# )

# Launch the component
launcher = Launcher()
launcher.add_pkg(components=[px4])
launcher.bringup()