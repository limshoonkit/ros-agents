from agents.components import Vision, MLLM
from agents.models import VisionModel, Idefics2, Llava
from agents.clients.roboml import RESPModelClient, HTTPModelClient
from agents.clients.ollama import OllamaClient
from agents.ros import Topic, Launcher
from agents.config import VisionConfig
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# Create QoS profile
qos_profile = {
    'reliability': ReliabilityPolicy.BEST_EFFORT,
    'durability': DurabilityPolicy.VOLATILE,
    'history': HistoryPolicy.KEEP_LAST,
}

# Create topics with the QoS profile
image0 = Topic(name="/camera/image", msg_type="Image", qos_profile=qos_profile)
detections_topic = Topic(name="detections", msg_type="Detections")

object_detection = VisionModel(
    name="object_detection", checkpoint="dino-4scale_r50_8xb2-12e_coco"
)
roboml_detection = RESPModelClient(object_detection)

detection_config = VisionConfig(threshold=0.5)
vision = Vision(
    inputs=[image0],
    outputs=[detections_topic],
    trigger=image0,
    config=detection_config,
    model_client=roboml_detection,
    component_name="detection_component",
)

text_query = Topic(name="text0", msg_type="String")
text_answer = Topic(name="text1", msg_type="String")

# idefics = Idefics2(name="idefics_model")
# idefics_client = HTTPModelClient(idefics, port=9000)

llava = Llava(name="llava")
llava_client = OllamaClient(llava)

mllm = MLLM(
    inputs=[text_query, image0, detections_topic],
    outputs=[text_answer],
    model_client=llava_client,
    trigger=text_query,
    component_name="mllm_component",
)

# mllm.set_component_prompt(
#     template="""Imagine you are a robot.
#     This image has following items: {{ detections }}.
#     Answer the following about this image: {{ text0 }}"""
# )

mllm.set_component_prompt(
    template="""
    Imagine you are a robot of size 1m x 1m x 1m equipped with a camera. 
    You are only allowed to use the following commands:
        turn(angle): turn the robot by a given number of degrees
        move(distance): moves the robot straight forward by a given distance in meters.
    You should reply with only one command at a time. The distance is in meters, and the direction an angle in degrees with respect to the robot's orientation.
    Negative angles are to the left and positive angles are to the right.  
    You should avoid running into any objects.
    """
)


launcher = Launcher()
launcher.add_pkg(components=[vision, mllm])
launcher.bringup()
