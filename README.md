# Surveillance-Camera-Face-Detection
*Last Update: 5 January, 2025*

### *Create by*
* Rega Arzula Akbar
* Muhammad Hafiz
* Bustan Nabiel Maulana
* Fahri Fahrezi
* Ridho Rizky Razami

### Introduction
Electrical Engineering Microcomputer Course Assignment
This project demonstrates how to stream live video using Picamera2 with MJPEG format. The application serves the stream over HTTP, accessible from a browser.

### Requirements
```

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
