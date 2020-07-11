
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            # Initialize the plugin
            self.ie = IECore()
            # Read model from IR and creates an IENetwork object
            self.model = self.ie.read_network(self.model_structure, self.model_weights)             
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        self.exec_network = None        

    def load_model(self, device):
        # Load the network `self.model` that was read from IR to the plugin `self.ie` with specified device name        
        # and creates an ExecutableNetwork object of the IENetwork class
        self.exec_network = self.ie.load_network(self.model, device)
        
        return   
        
    def predict(self, image):
        # Resize and reshape an image for inference
        image4infer = self.preprocess_input(image)
        # Start asynchronous inference       
        self.exec_network.start_async(request_id=0, inputs={self.input_name: image4infer}) 
        # Wait for async request to be complete and than extract outputs
        if self.exec_network.requests[0].wait(-1) == 0:
            output = self.exec_network.requests[0].outputs[self.output_name]
            
        coords = self.preprocess_outputs(output, image) # return coordinates of bboxes in the frame
        
        image = self.draw_outputs(coords, image)  # return frame copy with drawn bboxes
            
        return coords, image
    
    def draw_outputs(self, coords, image):
        image_copy = image.copy()
        for coord in coords:
            cv2.rectangle(image_copy, coord[:2], coord[2:], (255,0,0), 2 )
            
        return image_copy

    def preprocess_outputs(self, outputs, image):
        height, width = image.shape[0:2]
        coords = []
        for bbox in outputs[0][0]:
            if bbox[2] >= self.threshold:
                xmin = int(bbox[3] * width)
                ymin = int(bbox[4] * height)
                xmax = int(bbox[5] * width)
                ymax = int(bbox[6] * height)
                coords.append((xmin, ymin, xmax, ymax)) # append coordinates of bounding box to coords list
        
        return coords

    def preprocess_input(self, image):
        # get the model input shape
        model_shape = self.input_shape
        model_w = model_shape[3]
        model_h = model_shape[2]
        # copy the frame as numpy.ndarray and assignes the returning copy to the frame4infer variable.
        frame4infer = np.copy(image)
        frame4infer = cv2.resize(frame4infer, (model_w, model_h))
        frame4infer = frame4infer.transpose((2,0,1))
        frame4infer = frame4infer.reshape(1, 3, model_h, model_w)

        return frame4infer 


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model(device)
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)
