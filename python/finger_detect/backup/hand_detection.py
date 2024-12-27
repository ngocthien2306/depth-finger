# import time
import numpy as np
import cv2
import os
import onnxruntime as ort
import torch
from typing import List
from time import time
from time import sleep
import torchvision
palette = np.array(
    [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)
kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
skeleton = [[1, 1]]
limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
def xy2wh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def non_max_suppression(outputs, conf_threshold, iou_threshold, nc):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]  # batch size
    nc = nc or (outputs.shape[1] - 4)  # number of classes
    nm = outputs.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = outputs[:, 4:mi].amax(1) > conf_threshold  # candidates

    # Settings
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    t = time()
    output = [torch.zeros((0, 6 + nm), device=outputs.device)] * bs
    for index, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = wh2xy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_threshold]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections

        output[index] = x[i]
        if (time() - t) > time_limit:
            break  # time limit exceeded

    return output




class HandDetector:
    def __init__(self, path_model, input_size, stride=32, cuda: bool = False):
        self.path_model = path_model
        self.input_size = input_size
        self.stride = stride
        self.cuda = cuda
        self.ort_session = self.load_model(cuda)
    
    def load_model(self, cuda: bool = False):
        if cuda:
            # return ort.InferenceSession(self.path_model, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
            return ort.InferenceSession(self.path_model, providers=["CPUExecutionProvider"])
        return ort.InferenceSession(self.path_model)

    def preprocess_image(self, frame, input_size, stride):
        image = frame.copy()
        shape = image.shape[:2]  # current shape [height, width]
        
        # Calculate the ratio to resize the image while maintaining the aspect ratio
        r = min(input_size / shape[0], input_size / shape[1])
        new_unpad = (int(shape[1] * r), int(shape[0] * r))  # new width, new height
        padding = (input_size - new_unpad[0], input_size - new_unpad[1])  # padding required
        
        # Calculate padding values
        pad_w = padding[0] // 2
        pad_h = padding[1] // 2
        
        # Resize the image
        image = cv2.resize(image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR)

        # Add border/padding
        image = cv2.copyMakeBorder(
            image, pad_h, padding[1] - pad_h, pad_w, padding[0] - pad_w, cv2.BORDER_CONSTANT
        )

        # Convert HWC to CHW, BGR to RGB
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image, dtype=np.float32)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize to [0, 1]

        return image

    def postprocess(self, outputs, image, shape, original_frame):

        finger_list = []
        for output in outputs:
            output = output.clone()

            box_output = output[:, :6]
            if len(output):
                kps_output = output[:, 6:].view(len(output),3, 3)
            else:
                kps_output = output[:, 6:]

            r = min(image.shape[2] / shape[0], image.shape[3] / shape[1])

            box_output[:, [0, 2]] -= (image.shape[3] - shape[1] * r) / 2  # x padding
            box_output[:, [1, 3]] -= (image.shape[2] - shape[0] * r) / 2  # y padding
            box_output[:, :4] /= r

            box_output[:, 0].clamp_(0, shape[1])  # x
            box_output[:, 1].clamp_(0, shape[0])  # y
            box_output[:, 2].clamp_(0, shape[1])  # x
            box_output[:, 3].clamp_(0, shape[0])  # y

            kps_output[..., 0] -= (image.shape[3] - shape[1] * r) / 2  # x padding
            kps_output[..., 1] -= (image.shape[2] - shape[0] * r) / 2  # y padding
            kps_output[..., 0] /= r
            kps_output[..., 1] /= r
            kps_output[..., 0].clamp_(0, shape[1])  # x
            kps_output[..., 1].clamp_(0, shape[0])  # y

            for box in box_output:
                box = box.cpu().numpy()
                x1, y1, x2, y2, score, index = box
                # cv2.rectangle(
                #     original_frame,
                #     (int(x1), int(y1)),
                #     (int(x2), int(y2)),
                #     (0, 255, 0),
                #     2,
                # )

            for kpt in reversed(kps_output):

                for i, k in enumerate(kpt[:1]):
                    color_k = [int(x) for x in kpt_color[i]]
                    x_coord, y_coord = k[0], k[1]

                    finger_list.append([int(x_coord), int(y_coord)])
                    if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                        if len(k) == 3:
                            conf = k[2]
                            if conf < 0.3:
                                continue
                        cv2.circle(
                            original_frame,
                            (int(x_coord), int(y_coord)),
                            5,
                            color_k,
                            -1,
                            lineType=cv2.LINE_AA,
                        )


        return original_frame, finger_list

    def __call__(self, frame):
        image = frame
        start_preprocess = time()
        image = self.preprocess_image(image, self.input_size, self.stride)
        end_preprocess = time()
        input_tensor = torch.from_numpy(image)

    

        # print(input_tensor.shape)
        start_inference = time()
        ort_inputs = {self.ort_session.get_inputs()[0].name: image}
        outputs = self.ort_session.run(None, ort_inputs)[0]
        outputs = torch.from_numpy(outputs)
        end_inference = time()

        start_postprocess = time()
#        outputs = non_max_suppression(outputs, 0.5, 0.5, 1)
        outputs = non_max_suppression(outputs, 0.25, 0.3, 1)
        frame, finger_list = self.postprocess(
            outputs, input_tensor, frame.shape[:2], original_frame=frame
        )
        end_postprocess = time()
        preprocess_time= (end_preprocess - start_preprocess)*1000
        inference_time= (end_inference - start_inference)*1000
        postprocess_time= (end_postprocess - start_postprocess)*1000
        print(
            f"preprocess: {preprocess_time:.2f} ms, inference : {inference_time:.2f} ms, postprocess : {postprocess_time:.2f} ms"
        )

        return frame, finger_list

def load_model(path=
               r"fingertip.onnx"):
    if not os.path.exists(path):
        print(f"{path} isn't exists")
        return
    print("Loaded model fingertip detection")
    return HandDetector(path, 320, 32,cuda=True)


if __name__ == "__main__":
    hand_detector = HandDetector(r"fingertip.onnx", 320, 32,cuda=True)
    video = cv2.VideoCapture(0)

    # Get video properties for the output video
    fps = video.get(cv2.CAP_PROP_FPS)  # Frame rate
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frame
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frame

 
    count = 0
    sum_of_time = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break  # Exit if no frame is captured

        # Process the image and get output
        start_time = time()
        output = hand_detector(frame)
        end_time = time()
        sum_of_time += (end_time - start_time)
        sleep(0.03)
        count += 1

        # Optionally, visualize the output (modify based on your model output)
        # Example: draw bounding boxes on the frame if your model outputs them
        # (Assuming 'output' contains bounding boxes)
        # for box in output:
        #     x1, y1, x2, y2, confidence = box  # Example unpacking
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box

    
        # Display the output frame (optional)
        cv2.imshow("Hand Detection", output)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    

    cv2.destroyAllWindows()
    print("Average time: ", (sum_of_time / count)*1000, "ms")