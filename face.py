
import asyncio
import cv2
from hume import HumeStreamClient, StreamSocket
from hume.models.config import FaceConfig

async def capture_and_save_image(file_path):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite(file_path, frame)
    cap.release()

async def capture_emotion():
    await capture_and_save_image("./extra/emotion.jpg")
    client = HumeStreamClient("api_key")
    config = FaceConfig(identify_faces=True)
    
    async with client.connect([config]) as socket:
        result = await socket.send_file("./extra/emotion.jpg")
        return result