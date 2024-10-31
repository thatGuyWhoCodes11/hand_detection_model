import socket
import numpy as np
from predict_hand import predict_hand
import base64
from io import BytesIO
from PIL import Image
import json
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind(("localhost",9090))
s.listen(10)
s.settimeout(3)
print("connected and ready!")
while True:
    try:
        conn, addr = s.accept()
        data=b''
        while True:
            part = conn.recv(4096)
            data+=part
            if "\n" in part.decode():
                break
        if not data:
            continue
        data = json.loads(data)
        frame=base64.b64decode(data["base64Image"])
        img=Image.open(BytesIO(frame))
        img=np.asarray(img)
        drawn_image,prediction,prediction_proba=predict_hand(img,data["arabic_mode"])
        drawn_image=Image.fromarray(drawn_image)
        buffer = BytesIO()
        drawn_image.save(buffer,format='PNG')
        base64Image=base64.b64encode(buffer.getvalue()).decode("ascii")
        data = {"base64Image":base64Image,"prediction":prediction,"prediction_proba":prediction_proba}
        data=json.dumps(data)
        conn.sendall(str.encode(data+"\n"))
    except socket.timeout:
        pass