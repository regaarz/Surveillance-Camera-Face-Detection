# Surveillance-Camera-Face-Detection
*Last Update: 5 January, 2025*

### *Create by*
* Rega Arzula Akbar
* Muhammad Hafiz
* Bustan Nabiel Maulana
* Fahri Fahrezi
* Ridho Rizky Razami

### Introduction
This project is a Raspberry Pi-based surveillance camera system equipped with special capabilities to meet the needs of surveillance and user interaction. The system uses a fisheye camera to capture multiple viewpoints in a single camera, effectively providing wide-area coverage.

## Key Features

1. Multi-View with Fisheye Camera
The fisheye camera is used to capture images from multiple viewpoints in a single capture, allowing:
*Extensive monitoring without the need for multiple cameras.
*Multi-view display that is visualized directly on the user interface.

2. Face Detection and Hand Gesture
*The system is capable of detecting faces for various purposes such as:
*Automatic attendance through face recognition.
*Hand gestures are used to perform interactions, such as confirming presence with certain gestures.
3. User Interface
*The laptop or computer-based interface displays only the control menu and visualization results from the camera.
*The menu on the interface includes:
Multi-view display of the fisheye camera.
Face registration, where users can add new face data for attendance purposes.
Face and hand gesture detection monitoring.
4. Raspberry Pi-based Process
*All camera processing (face and gesture detection) is done directly on the Raspberry Pi, thus:
*Laptop or computer only serves as display media (not burdened by heavy processing).
*The system remains efficient and power efficient.

Translated with DeepL.com (free version)

### Requirements
```
bentar
```

### How to Run in Raspberry Pi| Usage
To use this system, follow these steps:
1. Prepare the Raspberry Pi, Camera Module
2. Open the Surveillance directory in the terminal
3. Update system Raspberry Pi
```
$ sudo raspi-config
```
4. Install library
```
$ sudo apt install -y python3-picamera2 python3-opencv
```
5. Run the code 
```
$ python3 main.py
```
6. Access the stream
```
$ http://<raspberrypi.local:8000
```

### How to Run in Laptop| Usage
To use Surveillance Camera, follow these steps :
1. Open the Surveillance directory in the terminal.
2. Build the virtual environment:
```
$ virtualenv venv
```
3. Activated the virtual environment:
```
$ source venv/bin/activate
```
4. Navigate to the plugin directory:
```
$ cd src/surveillance
```
5. Clone the repository :
```
$ git clone https://github.com/regaarz/Surveillance-Camera-Face-Detection.git
```
5. Navigate to car parking plugin:
```
$ cd moilapp-plugin-car-parking-systems
```
8. Install requirements:
```
$ pip install -r requirements.txt
```
9. Back to the *src* directory:
```
$ cd ../../
```
10. Run the Moilapp:
```
$ python3 main.py

   
### Contact
For any questions, suggestions, or concerns regarding the Plugin Application, please feel free to contact the repository owner 
