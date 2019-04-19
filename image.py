import cv2
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
from multiprocessing import Process
import starter as start
class recording_device():

    def __init__(self, model_used):
        self.model_used = model_used


    def counter(self):

        now = datetime.now()
        then = datetime.now() + timedelta(seconds=5.05)
        fl = 0
        while(datetime.now() < then):
            sec = datetime.now() - now
            if sec.seconds != fl:
                print(fl)
                fl = sec.seconds

    def image_record(self):

        now = datetime.now()
        then = datetime.now() + timedelta(seconds=5.05)
        fl = 0
        cap = cv2.VideoCapture(0)
        while(datetime.now() < then):

            # Capture frame-by-frame

            ret, frame = cap.read()

            # Our operations on the frame come here
            ret, jpeg = cv2.imencode('.jpg', frame)
            # Display the resulting frame
            cv2.imshow('',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        frame = cv2.resize(frame, (64,64))
        self.frame = frame.reshape(1,3,64,64)
        return self.frame

    def modeler(self):
        if self.model_used == 'CNN':
            self.model = start.open_CNN()
        else:
            self.network = start.incep_network()
            self.model = start.random_19()
            self.labels = start.labeler()
        return self.model
    def scaler(self):
        self.scaler = start.reload_scaler()
    def predict(self):
        rd.image_record()
        if self.model_used == 'CNN':
            return start.cnn_predict(self.model, self.frame)
        else:
            self.frame = self.frame.reshape(1,64,64,3)
            flat_map = self.network.predict(self.frame).reshape(1,2048)
            pred_gen = self.model.predict_proba(flat_map)
            arr_sort = pred_gen.argsort()[0][-3:][::-1]
            return [self.labels[t] for t in arr_sort], pred_gen[0][arr_sort]

if __name__=="__main__":
    rd = recording_device(model_used = 'Pre-Trained')
    rd.modeler()
