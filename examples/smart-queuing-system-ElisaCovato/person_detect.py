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
        self.queues = []

    # Points on frame where to look for queue
    def add_queue(self, points):
        self.queues.append(points)

    # Get part of the frame defined by queue points
    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max = q
            frame = image[y_min:y_max, x_min:x_max]
            yield frame

    # Check that coords from the image detection
    # are inside queue frame
    def check_coords(self, coords):
        d = {k + 1: 0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0] > q[0] and coord[2] < q[2]:
                    d[i + 1] += 1
        return d


class PersonDetect:
    '''
        Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold

        # Initialize the network
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path?")

        self.input_blob = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_blob].shape
        self.output_blob = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_blob].shape

    # Function to load the model in the network
    def load_model(self):
        core = IECore()
        self.net = core.load_network(network=self.model, device_name=self.device, num_requests=1)

    # Function to detect people in the frame:
    # get a frame from the video, runs the inference, returns the boxes
    # where people have been detected
    def predict(self, image):
        # Create input image to feed into the network
        net_input = {self.input_blob: self.preprocess_input(image)}
        # Start sync inference
        start = time.time()
        self.infer_request_handle = self.net.start_async(request_id=0, inputs=net_input)
        # Wait for the result
        if self.infer_request_handle.wait() == 0:
            # Get result of the inference request
            infer_time = time.time() - start
            outputs = self.infer_request_handle.outputs[self.output_blob]
            coords = self.preprocess_outputs(outputs)
            boxes, image = self.draw_outputs(coords, image)
            # Write inference time info on output video
            inf_time_message = "Inference time: {:.3f}ms".format(infer_time * 1000)
            cv2.putText(image, inf_time_message, (15, 85), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 153, 153), 2)
            return boxes, image

    # Given a frame from the video and the coords of detections,
    # this function draws boxes on the image.
    def draw_outputs(self, coords, image):
        w = image.shape[1]
        h = image.shape[0]
        boxes = []
        for coord in coords:
            xmin = int(coord[0] * w)
            ymin = int(coord[1] * h)
            xmax = int(coord[2] * w)
            ymax = int(coord[3] * h)
            boxes.append([xmin, ymin, xmax, ymax])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (145, 50, 255), 2)
        return boxes, image

    # If detection probability is higher than threshold probability,
    # the function returns the coords of the detection
    def preprocess_outputs(self, outputs):
        coords = []
        for obj in outputs[0][0]:
            if obj[2] > self.threshold:
                coords.append(obj[3:])
        return coords

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image


def main(args):
    model = args.model
    device = args.device
    video_file = args.video
    max_people = args.max_people
    threshold = args.threshold
    output_path = args.output_path

    start_model_load_time = time.time()
    pd = PersonDetect(model, device, threshold)
    print("Loading the model")
    pd.load_model()
    print("Model Loaded")
    total_model_load_time = time.time() - start_model_load_time
    print("Total model load time: ", total_model_load_time, "ms")

    queue = Queue()

    try:
        queue_param = np.load(args.queue_param)
        print("Adding queue params.")
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("Error loading queue param file")

    try:
        cap = cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: " + video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps,
                                (initial_w, initial_h), True)

    counter = 0
    start_inference_time = time.time()
    print("Starting inference")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1

            coords, image = pd.predict(frame)
            # Count how many people are inside each queue boxes
            num_people = queue.check_coords(coords)
            print(f"Total people on frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text = ""
            y_pixel = 25

            # Add tot number of people on screen message
            cv2.putText(image, f"Total people on screen: {len(coords)}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 124, 124), 2)

            # Create an overlay of the original image
            overlay_boxes = image.copy()
            # For each queue box, draw it on the original image, add text and other info
            for idx, q in enumerate(queue.queues):
                # Add queue box
                overlay_boxes = cv2.rectangle(overlay_boxes, (q[0], q[1]), (q[2], q[3]), (0, 255, 0), 5)
                # Add text box (queue id and num people in that queue) inside queue box
                cv2.rectangle(overlay_boxes, (q[0], q[1]), (q[0] + 240, q[1] + 90), (0, 0, 0), -1)
                cv2.putText(overlay_boxes, f"Queue ID: {idx + 1}", (q[0] + 5, q[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2,(0, 255, 0), 4)
                if idx == 0:
                    people = num_people[1]
                elif idx == 1:
                    people = num_people[2]
                out_text += f"People: {people}"
                cv2.putText(overlay_boxes, out_text, (q[0] + 5, q[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if people >= int(max_people):
                    msg = f"Queue full: please move to next queue"
                    cv2.rectangle(overlay_boxes, (q[0], q[1] + 100), (q[0] + 650, q[1] + 150), (0, 0, 0), -1)
                    cv2.putText(overlay_boxes, msg, (q[0] + 5, q[1] + 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                out_text = ""

            # Add overlay to original image
            image = cv2.addWeighted(overlay_boxes, 0.7, image, 0.3, 0)
            image = cv2.resize(image, (initial_w, initial_h))
            out_video.write(image)

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter / total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str("Total inference time (s): ") + str(total_inference_time) + '\n')
            f.write(str("Frame per second: ") + str(fps) + '\n')
            f.write(str("Model loading time (s): ") + str(total_model_load_time) + '\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='./results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)

    args = parser.parse_args()

    main(args)
