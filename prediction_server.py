import socket
import numpy as np
from predict_hand import predict_hand
import base64
from io import BytesIO
from PIL import Image
import json
import cv2
from threading import Thread

def conn_thread(conn):
    nextpart=None
    while True:
        clientdata=b''
        if nextpart:
            clientdata+=nextpart
        while True:
            try:
                part = conn.recv(4096)
            except ConnectionError:
                conn.close()
                return
            if not part:
                conn.close()
                return
            if b'*' in part:
                part,nextpart = part.split(b'*')
                clientdata+=part
                break
            clientdata+=part
        if not clientdata:
            conn.close()
            return
        try:
            data =json.loads(clientdata.decode("utf-8"))
            frame=base64.b64decode(data["base64Image"])
            img=Image.open(BytesIO(frame))
            img=np.asarray(img)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            drawn_image,prediction,prediction_proba=predict_hand(img,data["arabic_mode"])
            drawn_image=cv2.cvtColor(drawn_image,cv2.COLOR_RGB2BGR)
            drawn_image=Image.fromarray(drawn_image)
            buffer = BytesIO()
            drawn_image.save(buffer,format='JPEG')
            base64Image=base64.b64encode(buffer.getvalue()).decode("ascii")
            data = {"base64Image":base64Image,"prediction":prediction,"prediction_proba":prediction_proba}
            data=json.dumps(data)
            conn.sendall(str.encode(data+"\n"))
        except ConnectionError:
            pass
        

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind(("192.168.0.144",9090))
s.listen(10)
s.settimeout(None)
print("Server has started")
while True:
    try:
        conn, addr = s.accept()
        print("Socket Connection Accepted")
        Thread(target=conn_thread,args=(conn,)).start()
    except socket.timeout:
        pass