import sys
from PyQt6.QtWidgets import (
    QWidget, QApplication, QTableWidgetItem, QDialog, QInputDialog,
    QHeaderView, QTreeWidgetItem, QMenu, QMessageBox,
    QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QLineEdit
)
from PyQt6.QtCore import QTimer, Qt, QEvent, QSize
from PyQt6.QtGui import QIcon, QPixmap, QImage
from ui_main import Ui_Form
import mediapipe as mp
import cv2
import os
from datetime import datetime
import pickle       
from sklearn import svm
import face_recognition
import random
from qt_material import apply_stylesheet

def load_model(path='model/face_recognition_model.pkl'):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Model not found. Ensure the path is correct.")
        sys.exit(1)

def save_face(name, frame):
    dataset_dir = "dataset"
    person_dir = os.path.join(dataset_dir, name)
    
    os.makedirs(person_dir, exist_ok=True)
    
    image_count = len([f for f in os.listdir(person_dir) if f.endswith('.jpg')])

    image_path = os.path.join(person_dir, f"{name}_{image_count}.jpg")
    cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return image_path

def extract_features(image_path):
    """
    Extract face encodings using MediaPipe's face detection.
    """
    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_results = mp_face_detection.process(rgb_image)

    features = []
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = rgb_image.shape
            x_min = int(bbox.xmin * iw)
            y_min = int(bbox.ymin * ih)
            x_max = x_min + int(bbox.width * iw)
            y_max = y_min + int(bbox.height * ih)

            face_encodings = face_recognition.face_encodings(rgb_image, [(y_min, x_max, y_max, x_min)])
            if face_encodings:
                features.append(face_encodings[0])

    return features

def prepare_dataset(dataset_path):
    """
    Loads images from the dataset folder, extracts features, and collects labels.
    """
    features, labels = [], []
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            face_encodings = extract_features(image_path)
            if face_encodings:
                features.extend(face_encodings)
                labels.extend([person_name] * len(face_encodings))
                print(f"Processed {image_path}: Found {len(face_encodings)} face(s)")
    return features, labels

def train_model(features, labels, model_save_path):
    """
    Train the SVM model using features and labels, then save the model.
    """
    if len(set(labels)) <= 1:
        raise ValueError("The number of classes must be greater than one; only one class detected.")

    clf = svm.SVC(gamma='scale')
    clf.fit(features, labels)
    with open(model_save_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model trained and saved to {model_save_path}")

class AddFaceDialog(QDialog):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Face")
        self.resize(400, 300)

        self.frame = frame
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.image_label = QLabel(self)
        self.update_frame(self.frame)
        layout.addWidget(self.image_label)
        
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Enter Name")
        layout.addWidget(self.name_input)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def update_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def get_name(self):
        return self.name_input.text().strip()

def count_fingers(hand_landmarks, hand_label):
    
    fingers_up = 0
    hand_landmark = mp.solutions.hands.HandLandmark

    
    if hand_label != "Right":
        if hand_landmarks.landmark[hand_landmark.THUMB_TIP].x > hand_landmarks.landmark[hand_landmark.THUMB_IP].x:
            fingers_up += 1
    else:  
        if hand_landmarks.landmark[hand_landmark.THUMB_TIP].x < hand_landmarks.landmark[hand_landmark.THUMB_IP].x:
            fingers_up += 1
    
    finger_tips = [
        hand_landmark.INDEX_FINGER_TIP,
        hand_landmark.MIDDLE_FINGER_TIP,
        hand_landmark.RING_FINGER_TIP,
        hand_landmark.PINKY_TIP
    ]

    finger_pips = [
        hand_landmark.INDEX_FINGER_PIP,
        hand_landmark.MIDDLE_FINGER_PIP,
        hand_landmark.RING_FINGER_PIP,
        hand_landmark.PINKY_PIP
    ]

    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers_up += 1

    return fingers_up

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        self.target_fingers = random.randint(0, 5)

        self.ui.startButton.clicked.connect(self.toggle_camera)
        self.ui.addButton.clicked.connect(self.add_new_face)

        self.ui.tableWidget.setColumnCount(4)
        self.ui.tableWidget.setHorizontalHeaderLabels(["Name", "Fingers", "Date", "Time"])
        header = self.ui.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.ui.treeWidget.setColumnCount(1)
        self.ui.treeWidget.setHeaderLabels(["Dataset"])
        self.ui.treeWidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.treeWidget.customContextMenuRequested.connect(self.open_context_menu)
        self.ui.treeWidget.setIconSize(QSize(64, 64))  
        self.ui.treeWidget.viewport().installEventFilter(self)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        self.camera = cv2.VideoCapture('http://raspberrypi.local:8000/stream.mjpg')
        self.camera_running = True
        self.timer.start()
        self.frame = None

        self.finger_timer = QTimer(self)
        self.finger_timer.timeout.connect(self.update_random_fingers)
        self.finger_timer.start(5000) 
        
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.load_tree("dataset")
        self.clear_preview()

        self.model = load_model()

    def update_random_fingers(self):
        """ Update target fingers every 5 seconds """
        self.target_fingers = random.randint(0, 5)
        print(f"New Target Fingers: {self.target_fingers}")

    def load_tree(self, root_path):
        if not os.path.exists(root_path):
            return
        
        self.ui.treeWidget.clear()
        
        root_item = QTreeWidgetItem(self.ui.treeWidget, [os.path.basename(root_path)])

        for folder in os.listdir(root_path):
            folder_path = os.path.join(root_path, folder)
            if os.path.isdir(folder_path):
                folder_item = QTreeWidgetItem(root_item, [folder])

                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        file_item = QTreeWidgetItem(folder_item, [file])
                        file_item.setIcon(0, QIcon(file_path))
                        file_item.setData(0, Qt.ItemDataRole.UserRole, file_path)

        self.ui.treeWidget.expandAll()

    def open_context_menu(self, position):
        selected_item = self.ui.treeWidget.itemAt(position)
        if selected_item:
            context_menu = QMenu(self)
            rename_action = context_menu.addAction("Rename")
            delete_action = context_menu.addAction("Delete")

            action = context_menu.exec(self.ui.treeWidget.viewport().mapToGlobal(position))
            if action == delete_action:
                self.delete_item(selected_item)
            elif action == rename_action:
                self.rename_item(selected_item)

    def rename_item(self, item):
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        
        if not file_path or not os.path.isfile(file_path):
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid file.")
            return

        parent_folder = os.path.dirname(file_path)
        old_folder_name = os.path.basename(parent_folder)
        
        new_folder_name, ok = QInputDialog.getText(self, "Rename", f"Enter new name for '{old_folder_name}':")
        if not ok or not new_folder_name.strip():
            return
        
        try:
            self.rename_folder_files(parent_folder, new_folder_name)
            new_folder_path = os.path.join(os.path.dirname(parent_folder), new_folder_name)
            os.rename(parent_folder, new_folder_path)
            QMessageBox.information(self, "Success", f"Renamed folder and files to '{new_folder_name}'")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to rename: {str(e)}")

        self.load_tree("dataset")

    def rename_folder_files(self, folder_path, new_folder_name):
        """
        Renames all .jpg files in the specified folder using the new folder name.
        """
        files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
        )

        for idx, old_file_name in enumerate(files):
            old_file_path = os.path.join(folder_path, old_file_name)
            new_file_name = f"{new_folder_name}_{idx}.jpg"
            new_file_path = os.path.join(folder_path, new_file_name)

            if old_file_path != new_file_path:
                os.rename(old_file_path, new_file_path)

    def delete_item(self, item):
        parent = item.parent()
        file_path = item.data(0, Qt.ItemDataRole.UserRole)

        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            QMessageBox.information(self, "Deleted", f"{file_path} deleted.")
            
            
            folder_path = os.path.dirname(file_path)
            self.rename_sibling_files(folder_path)

            
            self.load_tree("dataset")
        elif parent:
            parent.removeChild(item)
        else:
            self.ui.treeWidget.takeTopLevelItem(self.ui.treeWidget.indexOfTopLevelItem(item))

    def rename_sibling_files(self, folder_path):
        """
        Renames files in the given folder to ensure sequential numbering.
        """
        files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
        )

        for idx, file_name in enumerate(files):
            old_file_path = os.path.join(folder_path, file_name)
            new_file_name = f"{os.path.basename(folder_path)}_{idx}.jpg"
            new_file_path = os.path.join(folder_path, new_file_name)

            
            if old_file_path != new_file_path:
                os.rename(old_file_path, new_file_path)

    def eventFilter(self, source, event):
        if source == self.ui.treeWidget.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            pos = event.pos()
            item = self.ui.treeWidget.itemAt(pos)
            if item:
                self.show_image_preview(item)
            else:
                self.clear_preview()
            return True
        return super().eventFilter(source, event)

    def show_image_preview(self, item):
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        if file_path and os.path.isfile(file_path):
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.ui.preview_label.setPixmap(pixmap)
                self.ui.preview_label.setText("")
                return
        self.clear_preview()

    def clear_preview(self):
        self.ui.preview_label.setText("Click on a file to preview")

    def toggle_camera(self):
        """
        Start or stop the camera feed.
        """
        if self.camera_running:
            self.timer.stop()
            self.camera.release()
            self.ui.startButton.setText("Start Camera")
            self.ui.displayCamera.clear()
            self.camera_running = False
        else:
            self.camera.open(0)  
            self.timer.start(15)  
            self.ui.startButton.setText("Stop Camera")
            self.camera_running = True

    def update_frame(self):
        ret, frame = self.camera.read()
        self.frame = frame
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = self.mp_face_detection.process(rgb_frame)
        predictions = self.handle_face_detection(frame, rgb_frame, face_results)
        
        hand_results = self.mp_hands.process(rgb_frame)
        self.handle_hand_detection(rgb_frame, hand_results, predictions)

        cv2.putText(rgb_frame, f"Show {self.target_fingers} fingers!", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        height, width, _ = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.ui.displayCamera.setPixmap(QPixmap.fromImage(q_image))

    def handle_face_detection(self, frame, rgb_frame, face_results):
        predictions = []
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x_min = int(bbox.xmin * iw)
                y_min = int(bbox.ymin * ih)
                x_max = x_min + int(bbox.width * iw)
                y_max = y_min + int(bbox.height * ih)
                face_encodings = face_recognition.face_encodings(rgb_frame, [(y_min, x_max, y_max, x_min)])
                predictions = [self.model.predict([encoding])[0] for encoding in face_encodings]
                for name in predictions:
                    cv2.rectangle(rgb_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(rgb_frame, name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return predictions

    def handle_hand_detection(self, rgb_frame, hand_results, predictions):
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                label = hand_results.multi_handedness[idx].classification[0].label
                fingers = count_fingers(hand_landmarks, label)
                label_text = "Right" if label == "Left" else "Left"
                if fingers == self.target_fingers:
                    cv2.putText(rgb_frame, f"{label_text}: {fingers} fingers (Match!)", (50, 100 + idx * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    for name in predictions:
                        self.add_or_update_table(name, fingers)
                else:                    
                    cv2.putText(rgb_frame, f"{label_text}: {fingers} fingers", (50, 100 + idx * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    def add_new_face(self):
        if self.frame is None:
            QMessageBox.critical(self, "Error", "Unable to capture the frame.")
            return
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        dialog = AddFaceDialog(rgb_frame, self)
        if dialog.exec():  
            name = dialog.get_name()
            if name:
                saved_path = save_face(name, rgb_frame)
                QMessageBox.information(self, "Success", f"{name} saved at {saved_path}")
                self.load_tree("dataset")  
            else:
                QMessageBox.warning(self, "Warning", "Name cannot be empty!")
        
        dataset_path = 'dataset'
        model_save_path = "model/face_recognition_model.pkl"
        features, labels = prepare_dataset(dataset_path)

        if features and labels:
            print("Training the model...")
            train_model(features, labels, model_save_path)
            self.model = load_model()
        else:
            print("No valid face encodings found in the dataset.")

    def add_or_update_table(self, name, fingers):
        """
        Adds or updates the detected person's info in the table.
        - Updates 'Fingers' if the person already exists.
        - Adds a new entry if the person is not found.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")        
        row_count = self.ui.tableWidget.rowCount()
        for row in range(row_count):
            table_name = self.ui.tableWidget.item(row, 0).text()
            if table_name == name:
                
                self.ui.tableWidget.setItem(row, 1, QTableWidgetItem(str(fingers)))
                self.ui.tableWidget.setItem(row, 2, QTableWidgetItem(now.split(" ")[0]))  
                self.ui.tableWidget.setItem(row, 3, QTableWidgetItem(now.split(" ")[1]))  
                return
        self.ui.tableWidget.insertRow(row_count)
        self.ui.tableWidget.setItem(row_count, 0, QTableWidgetItem(name))
        self.ui.tableWidget.setItem(row_count, 1, QTableWidgetItem(str(fingers)))
        self.ui.tableWidget.setItem(row_count, 2, QTableWidgetItem(now.split(" ")[0]))  
        self.ui.tableWidget.setItem(row_count, 3, QTableWidgetItem(now.split(" ")[1]))  

    def closeEvent(self, event):
        """
        Ensure resources are released when the window is closed.
        """
        self.timer.stop()
        self.camera.release()
        event.accept()

app = QApplication(sys.argv)
apply_stylesheet(app, theme='dark_teal.xml')
widget = Widget()
widget.show()
app.exec()
