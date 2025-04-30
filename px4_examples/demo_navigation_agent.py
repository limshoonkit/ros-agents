import numpy as np

from typing import Optional
from agents.components import (
    MLLM, LLM
)
from agents.clients.roboml import HTTPModelClient
from agents.clients.ollama import OllamaClient
from agents.models import OllamaModel
from agents.config import MLLMConfig, LLMConfig
from agents.ros import Topic, Launcher

from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


### Setup our models###
# llava = Llava(name="llava", quantization="4bit", checkpoint="liuhaotian/llava-v1.6-mistral-13b") # 7b, 13b
# llama_vision = OllamaModel(name='llama_vision', checkpoint="lla3ma3.2-vision:11b-instruct-q8_0") # 11b-instruct-q8_0, 11b-instruct-q4_K_M
# moondream = OllamaModel(name='moondream', checkpoint="moondream:1.8b-v2-fp16") # moondream:1.8b-v2-fp16
# minicpm = OllamaModel(name='minicpm', checkpoint="minicpm-v:8b") # minicpm-v:8b
gemma = OllamaModel(name='gemma', checkpoint="gemma3:12b") # 1b, 4b, 12b

ollama_client = OllamaClient(gemma)

### Setup topics ###
qos_profile = {
    'reliability': ReliabilityPolicy.BEST_EFFORT,
    'durability': DurabilityPolicy.VOLATILE,
    'history': HistoryPolicy.KEEP_LAST,
}

query_topic = Topic(name="text0", msg_type="String")
introspection_query = Topic(name="introspection", msg_type="String", qos_profile=qos_profile)
query_answer = Topic(name="text1", msg_type="String")
introspection_answer = Topic(name="introspection_answer", msg_type="String")
image0 = Topic(name="zed/left/rgb", msg_type="Image", qos_profile=qos_profile)
image1 = Topic(name="cam0/rgb", msg_type="Image", qos_profile=qos_profile)

### Setup components ###
vln_agent_config = MLLMConfig(
    # enable_rag=True,
    # collection_name="map",
    # distance_func="l2",
    # n_results=1,
    # add_metadata=True,
    chat_history=False,
    history_size=10,
    temperature=0.7,
    max_new_tokens=10,
)

vln_agent = MLLM(
    inputs=[query_topic, image0],
    outputs=[query_answer],
    model_client=ollama_client,  # http_client, ollama_client
    trigger=query_topic,
    component_name="vln_agent",
    config=vln_agent_config
)

# vln_agent.set_component_prompt(
#   template="""
#   ROLE:
#   - You are a competent drone navigation system that can guide the drone to the designated goal.
 
#   COMMAND TYPES:
#   - `Turn(angle)`: Rotate the drone (negative = left, positive = right)
#   - `Move(distance)`: Move the drone (negative = backward, positive = forward)
  
#   NAVIGATION RULES:
#     - Use only turn magnitude up to 90 degrees
#     - Use only movement distance up to 2.0 meters
#     - Always prioritize avoiding any obstacles (netted enclosure, cardboard boxes, etc.)
#     - You are in a 4.5m x 7m netted enclosure and should *NOT* exit it. 
#     - You should only response with a *SINGLE* command at a time without any further explanation.
#     - You are provided with distances to the goal, current position, current heading, goal position, obstacles position and previous commands at each step.
#     - X-axis is left/right, Y-axis is forward/backward

#   COMMAND EXAMPLES:
#     - Turn(45)
#     - Move(1.5)
#     - Turn(-30)
#     - Move(0.3)
#   """
# )

vln_agent.set_component_prompt(
   template="""
    ROLE:
    - You are a competent drone navigation system that can guide the drone to the designated goal.

    INPUT DATA FORMAT:
    - {current_position}: (x, y) coordinates in meters
    - {current_heading}: angle in degrees (0 = facing +Y direction)
    - {goal_position}: (x, y) coordinates in meters
    - {distance_to_goal}: direct distance in meters
    - {obstacles}: list of obstacle positions [(x, y), ... ]
    - {previous_commands}: list of commands executed

    COMMAND TYPES:
    - Turn(angle): Rotate the drone (negative = left, positive = right)
    - Move(distance): Move the drone (negative = backward, positive = forward)

    NAVIGATION RULES:
    - You *MUST* use only turn magnitude up to 90 degrees
    - You *MUST* use only movement distance up to 2.0 meters
    - You *MUST* avoid moving through cardboard boxes obstacles.
    - You *MUST* stay within the netted enclosure.
    - You *MUST* only response with a *SINGLE* command at a time without any further explanation.
    - You *MUST* vary your commands and values based on the proximity to obstacles and goal. 
    
    NAVIGATION STRATEGY:
    - Calculate the angle to the goal relative to current heading
    - If angle to goal > 15 degrees: Turn toward goal first
    - If obstacle detected in path: Use smaller movements and adjust course
    - When close to goal (< 0.5m): Use smaller movements for precision
    - Alternate between turning and moving to make steady progress
    - Vary turn angles (use different angles between 10-90 degrees)
    - Vary movement distances (use different distances between 0.3-2.0 meters)

    COMMAND EXAMPLES:
    - When goal is far and clear path: Move(2.0)
    - When slight course correction needed: Turn(15)
    - When major course correction needed: Turn(75)
    - When approaching goal: Move(0.5)
    - When navigating around obstacle: Turn(45) or Turn(-45)
  """
)

# vln_agent.set_component_prompt(
#   template="""
#     ROLE:
#     You are a drone navigation module. You receive:
#     1. A front-facing camera image.
#     2. Numeric state vectors:
#         - current_position: (x, y) in meters
#         - current_heading: θ in degrees, 90 degree is facing +Y direction
#         - goal_position: (x_goal, y_goal) in meters
#         - obstacles_position: array of obstacle position [(x1, y1), (x2, y2), ...] in meters
#         - X-axis is left/right, Y-axis is forward/backward

#     TASK:
#     Compute exactly one low-level motion command to drive the drone toward the goal while avoiding obstacles.

#     COMMAND SYNTAX (output must be exactly one line, no extra text):
#     • Turn(angle)    — rotate in place by “angle” degrees (−90 ≤ angle ≤ +90)  
#     • Move(distance) — translate forward/backward by “distance” meters (−2.0 ≤ distance ≤ +2.0)

#     CONSTRAINTS:
#     • Only one command per observation.  
#     • Do not exceed ±90° for turns or ±2.0 m for moves.  
#     • If there are any obstacles within 0.5 m in front, use Turn to steer away from obstacles.  
#     • Otherwise, if the direct path to the goal is clear (all front obstatcles more than 0.5 m), use Move to advance.  
#     • Always stay within the netted enclosure.
#     • Do not repeat the same command with the same values twice in a row.

#     COMMAND EXAMPLES:
#     Turn(45)
#     Move(1.5)
#     Turn(-30)
#     Move(0.3)
#   """
# )

# Configure the introspector component
introspector_config = MLLMConfig(
    temperature=0.1,
    max_new_tokens=10,
)

# Initialize the introspector
introspector = MLLM(
    inputs=[introspection_query, image1],
    outputs=[introspection_answer],
    model_client=ollama_client, # http_client, ollama_client
    trigger=introspection_query,
    component_name="introspector",
    config=introspector_config
)

# Set a very specific prompt template for introspector
introspector.set_component_prompt(
    template="""
    You are a visual recognition system on a drone. Your task is to answer whether a specific object is visible in the current camera feed.

    IMPORTANT RULES:
    - Respond with EXACTLY ONE of the following (no other text, no explanation):
    - "Yes" → The object is clearly visible in the image.
    - "No" → The object is clearly NOT visible in the image.
    - "Invalid" → The input question does not relate to finding an object in image.

    ONLY respond with one of the above three words.
    """
)

def introspection_validation(output: str) -> Optional[str]:
    print(f"Raw output: {output}")
    for option in ["Invalid", "Yes", "No"]:
        if option in output.replace(' ', ''): # remove any trailing spaces
            return option

introspector.add_publisher_preprocessor(introspection_answer, introspection_validation)

# Launch the components
launcher = Launcher()
launcher.add_pkg(
    components=[
        vln_agent,
        introspector,
    ]
)
launcher.bringup()