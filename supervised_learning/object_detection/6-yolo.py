#!/usr/bin/env python3
"""0. Initialize Yolo"""
import numpy as np
import tensorflow as tf
import cv2
import os


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
        for box, conf, class_proba in zip(boxes,
                                          box_confidences, box_class_probs):
            scores = conf * np.max(class_proba, axis=-1, keepdims=True)
            filtering_mask = scores >= self.class_t
            filtered_boxes.append(box[filtering_mask[..., 0]])
            box_classes.append(np.argmax(class_proba[filtering_mask[..., 0]],
                                         axis=-1))
            box_scores.append(scores[filtering_mask[..., 0]])
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)
        box_scores = box_scores.flatten()
        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Return the filtered boxes, their classes, and scores in descending
        __order of confidence
        IoU = inters / union: Removes highly overlapping boxes during NMS."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []
        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """folder_path: a string representing the path to the folder holding
        __all the images to load
        Returns a tuple of (images, image_paths):
        images: a list of images as numpy.ndarrays
        image_paths: a list of paths to the individual images in images"""
        images = []
        image_paths = []
        for filename in os.listdir(folder_path):
            images_path = os.path.join(folder_path, filename)
            img = cv2.imread(images_path)
            if img is not None:
                images.append(img)
                image_paths.append(images_path)
        return images, image_paths

    def preprocess_images(self, images):
        """images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]
        Returns a tuple of (pimages, image_shapes):
        pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
        __containing all of the preprocessed images
        ni: the number of images that were preprocessed
        input_h: the input height for the Darknet model
        Note: this can vary by model
        input_w: the input width for the Darknet model
        Note: this can vary by model
        3: number of color channels
        image_shapes: a numpy.ndarray of shape (ni, 2)
        __containing the original height
        __and width of the images
        2 => (image_height, image_width)"""
        pimages = []
        image_shapes = []
        for img in images:
            img_h, img_w, img_c = img.shape
            image_shapes.append([img_h, img_w])
            # Resize the images with inter-cubic interpolation
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]
            resized_img = cv2.resize(img, dsize=(input_w, input_h),
                                     interpolation=cv2.INTER_CUBIC)
            # rescale to [0, 1]
            scaled_img = resized_img / 255.0
            pimages.append(scaled_img)
        # Convert lists to np.array
        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)
        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Displays image with all boundary boxes,class names,and box scores
        Args:
            image: a numpy.ndarray containing an unprocessed image
            boxes: a numpy.ndarray containing the boundary boxes for the image
            box_classes: numpy.ndarray containing class indices for each box
            box_scores: a numpy.ndarray containing the box scores for each box
            file_name: the file path where the original image is stored
        """
        # Draw bounding boxes, class names, and box scores on the image
        for box, cls, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = box.astype(int)
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Prepare the text to display
            class_name = self.class_names[cls]
            text = f"{class_name} {score:.2f}"
            # Put the text above the bounding box
            cv2.putText(image, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                            0, 0, 255), 1, cv2.LINE_AA)

        # Display the image
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        # Handle key presses
        if key == ord('s'):
            if not os.path.exists("detections"):
                os.makedirs("detections")
            save_path = os.path.join("detections", file_name)
            cv2.imwrite(save_path, image)
            print(f"Image saved to {save_path}")
        cv2.destroyAllWindows()
