import numpy as np

from typing import Optional
from agents.components import (
    MLLM, LLM
)
from agents.clients.roboml import HTTPModelClient
from agents.clients.ollama import OllamaClient
from agents.models import OllamaModel, Llava
from agents.config import MLLMConfig, LLMConfig
from agents.ros import Topic, Launcher

from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


### Setup our models###
# llava = Llava(name="llava", checkpoint="liuhaotian/llava-v1.6-mistral-7b") # 7b, 13b
# llama_vision = OllamaModel(name='llama_vision', checkpoint="llama3.2-vision:11b") # 11b-instruct-q8_0, 11b-instruct-q4_K_M
# moondream = OllamaModel(name='moondream', checkpoint="moondream:1.8b-v2-fp16") # moondream:1.8b-v2-fp16
# minicpm = OllamaModel(name='minicpm', checkpoint="minicpm-v:8b") # minicpm-v:8b
gemma = OllamaModel(name='gemma', checkpoint="gemma3:12b") # 4b, 12b
vlm_client = OllamaClient(gemma)

# qwen = OllamaModel(name='qwen', checkpoint="qwen2.5:3b") # 1.5b, 3b, 7b
# deepseek = OllamaModel(name='deepseek', checkpoint="deepseek-llm:7b") # not using reasoning model here
# llama = OllamaModel(name="llama", checkpoint="llama3.2:3b") # 1b, 3b
# llama = OllamaModel(name="llama, checkpoint="llama3.1:8b") # 
# gemma_llm = OllamaModel(name='gemma', checkpoint="gemma3:4b") # 1b, 4b, 12b
llm_client = OllamaClient(gemma)

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
embodied_agent_config = LLMConfig(
    # enable_rag=True,
    # collection_name="map",
    # distance_func="l2",
    # n_results=1,
    # add_metadata=True,
    chat_history=False,
    history_size=10,
    temperature=0.2,
    max_new_tokens=10,
)

embodied_agent = LLM(
    inputs=[query_topic],
    outputs=[query_answer],
    model_client=llm_client, 
    trigger=query_topic,
    component_name="embodied_agent",
    config=embodied_agent_config
)

# embodied_agent.set_system_prompt(
#     prompt="""
#     You are a navigation assistant for a drone equipped with a camera.
#     Your task is to guide the drone to reach a goal position at using simple motion commands.
    
#     COMMANDS:
#     - Turn(angle): Rotate the drone (angle in degrees, positive = right, negative = left).
#     - Move(distance): Move the drone (distance in meters).
    
#     RULES:
#     - *NEVER* output any natural language explanations.
#     - You *MUST* use only Move() and Turn() commands.
#     - You *MUST* use only turn magnitude up to 90 degrees
#     - You *MUST* use only movement distance up to 3.0 meters
#     - IMPORTANT: If you receive information that an object is NOT visible, you MUST ONLY output Turn() commands to search.
#     - IMPORTANT: If you receive information that an object IS visible, you can use Move() commands to approach the goal.
#     - Only output one command per step.
    
#     EXAMPLES:
#     - Turn(30)
#     - Move(2.5)
#     - Turn(-45)
#     - Move(0.5)
#     """
# )

embodied_agent.set_system_prompt(
    prompt="""
    You are a navigation assistant for a drone. On each step you will receive:

    • Current position: (x, y)  
    • Current heading: H° (–180…+180)  
    • Goal position: (gx, gy)  
    • Object detected: Yes or No  

    You must only output a *SINGLE* command line, in one of these two forms:

        Turn(Δ)  
        Move(d)  

    Where Δ is in degrees (–90…+90) and d is in meters (0.3…3.0).
    IMPORTANT: 
    Do *NOT* output any code fences, markdown, explanations, additional text.

    RULES:

    1. If Object detected: No → you MUST output only Turn(Δ).  
    2. If Object detected: Yes → you may output Move(d) or Turn(Δ).  

    To compute Δ when turning toward the goal:

    1. bearing = atan2(gy – y, gx – x) × (180/π)  
    2. raw Δ₀ = bearing – H  
    3. Δ = atan2(sin(Δ₀×π/180), cos(Δ₀×π/180)) × (180/π)  
    4. clip Δ into [–90, 90] and round to nearest integer  

    To compute d when moving toward the goal:

    1. distance = sqrt((gx – x)² + (gy – y)²)  
    2. clip d into [0.3, 3.0] and round to one decimal place

    EXAMPLES:
    ---
    Go towards the human at (-1.0, 1.5).
    Current position: (x = -0.011030, y = -0.074337). 
    Current heading: (90.092232). 
    Object detected: No

    Output: Turn(-45)
    ---
    Go towards the human at (-1.0, 1.5).
    Current position: (x = -0.12311, y = -0.053). 
    Current heading: (45.0123). 
    Object detected: Yes

    Output: Move(1.2)
    """
)

# Configure the introspector component
introspector_config = MLLMConfig(
    temperature=0.2,
    max_new_tokens=10,
)

# Initialize the introspector
introspector = MLLM(
    inputs=[introspection_query, image0],
    outputs=[introspection_answer],
    model_client=vlm_client,
    trigger=introspection_query,
    component_name="introspector",
    config=introspector_config
)

# Set a very specific prompt template for introspector
introspector.set_component_prompt(
    template="""
    You are a visual recognition system on a drone. Your task is to answer whether a humanoid robot is visible in the current camera feed.

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
        embodied_agent,
        introspector,
    ]
)
launcher.bringup()