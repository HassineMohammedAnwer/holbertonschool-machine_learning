
#!/usr/bin/env python3
"""0. Initialize Yolo"""
import numpy as np
import tensorflow as tf


class Yolo:
    """uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used for
        __the Darknet model, listed in order of index, can be found
        class_t is a float representing the box score threshold for the
        __initial filtering step
        nms_t is a float representing the IOU threshold for non-max suppression
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        __containing all of the anchor boxes:
        outputs is the number of outputs (predictions) made by Darknet model
        anchor_boxes is the number of anchor boxes used for each prediction
        2 => [anchor_box_width, anchor_box_height]
        Public instance attributes:
        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes"""
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = []
        with open(classes_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.class_names.append(line)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """outputs is a list of numpy.ndarrays containing the predictions
        __from the Darknet model for a single image:
        Each output will have the shape (grid_height, grid_width,
        anchor_boxes, 4 + 1 + classes)
        grid_height & grid_width => the height and width of the grid used
        __for the output
        anchor_boxes => the number of anchor boxes used
        4 => (t_x, t_y, t_w, t_h)
        1 => box_confidence
        classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image’s original size
        __[image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
        anchor_boxes, 4) containing the processed boundary boxes for each
        __output, respectively:
        4 => (x1, y1, x2, y2)
        (x1, y1, x2, y2) should represent the boundary box relative to
        __original image
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, 1) containing the box confidences for
        __each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, classes) containing the box’s class
        __probabilities for each output, respectively"""
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i_cell, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]
            image_height, image_width = image_size
            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            cx, cy = np.meshgrid(np.arange(grid_width),
                                 np.arange(grid_height))
            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)
            p_w = self.anchors[i_cell, :, 0]
            p_h = self.anchors[i_cell, :, 1]
            bx = (1.0 / (1.0 + np.exp(-t_x))) + cx
            by = (1.0 / (1.0 + np.exp(-t_y))) + cy
            bx /= grid_width
            by /= grid_height
            bw = p_w * np.exp(t_w)
            bw /= self.model.input.shape[1]
            bh = p_h * np.exp(t_h)
            bh /= self.model.input.shape[2]
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bw / 2 + bx) * image_width
            y2 = (bh / 2 + by) * image_height
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)
            Obj = output[:, :, :, 4:5]
            Objectness = 1 / (1 + np.exp(-Obj))
            Class_Confidences = output[:, :, :, 5:]
            sigmoid_class_probs = 1 / (1 + np.exp(-Class_Confidences))
            box_confidences.append(Objectness)
            box_class_probs.append(sigmoid_class_probs)
        return boxes, box_confidences, box_class_probs
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """filtering the bounding boxes based on confidence scores and class
        probabilities, and returning the relevant predictions.
        score=confidence × max(class_probabilities)"""
        filtered_boxes = []
        box_classes = []
        box_scores = []
        for box, conf, class_proba in zip(boxes, box_confidences, box_class_probs):
            scores = conf * np.max(class_proba, axis=-1, keepdims=True)
            filtering_mask = scores >= self.class_t
            filtered_boxes.append(box[filtering_mask[..., 0]])
            box_classes.append(np.argmax(class_proba[filtering_mask[..., 0]], axis=-1))
            box_scores.append(scores[filtering_mask[..., 0]])
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)
        return filtered_boxes, box_classes, box_scores
