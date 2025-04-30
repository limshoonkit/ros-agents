from typing import Optional
from agents.components import MapEncoding, Vision, MLLM
from agents.models import VisionModel, Llava
from agents.clients.roboml import RESPModelClient, HTTPDBClient
from agents.clients.ollama import OllamaClient
from agents.ros import Topic, MapLayer, Launcher, FixedInput
from agents.vectordbs import ChromaDB
from agents.config import MapConfig, VisionConfig
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

qos_profile = {
    'reliability': ReliabilityPolicy.BEST_EFFORT,
    'durability': DurabilityPolicy.VOLATILE,
    'history': HistoryPolicy.KEEP_LAST,
}

# Define the image input topic
image0 = Topic(name="camera/image", msg_type="Image", qos_profile=qos_profile)
introspection =  Topic(name="introspection", msg_type="String", qos_profile=qos_profile)
# Create a detection topic
detections_topic = Topic(name="detections", msg_type="Detections")

# Add an object detection model
object_detection = VisionModel(
    name="object_detection", checkpoint="dino-4scale_r50_8xb2-12e_coco"
)
roboml_detection = RESPModelClient(object_detection)

# Initialize the Vision component
detection_config = VisionConfig(threshold=0.5)
vision = Vision(
    inputs=[image0],
    outputs=[detections_topic],
    trigger=image0,
    config=detection_config,
    model_client=roboml_detection,
    component_name="detection_component",
)


# Define a model client (working with Ollama in this case)
llava = Llava(name="llava")
llava_client = OllamaClient(llava)

# Define a fixed input for the component
introspection_query = FixedInput(
    name="introspection_query",
    msg_type="String",
    fixed="What kind of a object is this? Is it a person, a pen or a book? Give a one word answer, out of the given choices",
)
# Define output of the component
introspection_answer = Topic(name="introspection_answer", msg_type="String")

# Start a timed (periodic) component using the mllm model defined earlier
# This component answers the same question after every 15 seconds
introspector = MLLM(
    inputs=[introspection, image0],  # we use the image0 topic defined earlier
    outputs=[introspection_answer],
    model_client=llava_client,
    trigger=introspection,  # we provide the time interval as a float value to the trigger parameter
    component_name="introspector",
)

introspector.set_component_prompt(
    template="""
    You are a visual recognition system on a drone. Your task is to answer whether a specific object is visible in the current camera feed.

    IMPORTANT RULES:
    - Respond with EXACTLY ONE of the following (no other text, no explanation):
    - "Yes" → The object is clearly visible in the image.
    - "No" → The object is clearly NOT visible in the image.
    - "Invalid" → The input question does not specify a valid, identifiable object (e.g., too vague or empty).

    ONLY respond with one of the above three words.
    """
)


# Define an arbitrary function to validate the output of the introspective component
# before publication.
def introspection_validation(output: str) -> Optional[str]:
    print(f"Raw output: {output}")
    for option in ["book", "pen", "person", "office"]:
        if option in output.lower():
            return option


introspector.add_publisher_preprocessor(introspection_answer, introspection_validation)

# # Object detection output from vision component
# layer1 = MapLayer(subscribes_to=detections_topic, temporal_change=True)
# # Introspection output from mllm component
# layer2 = MapLayer(subscribes_to=introspection_answer, resolution_multiple=3)

# # Initialize mandatory topics defining the robots localization in space
# position = Topic(name="odom", msg_type="Odometry")
# map_topic = Topic(name="map", msg_type="OccupancyGrid")

# # Initialize a vector DB that will store our semantic map
# chroma = ChromaDB(name="MainDB")
# chroma_client = HTTPDBClient(db=chroma)

# # Create the map component
# map_conf = MapConfig(map_name="map")  # We give our map a name
# map = MapEncoding(
#     layers=[layer1, layer2],
#     position=position,
#     map_topic=map_topic,
#     config=map_conf,
#     db_client=chroma_client,
#     trigger=15.0,
#     component_name="map_encoding",
# )

# Launch the components
launcher = Launcher()
# launcher.add_pkg(components=[vision, introspector, map])
launcher.add_pkg(components=[vision, introspector])
launcher.bringup()