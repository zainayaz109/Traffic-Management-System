
# Project Title

The traffic management system employs advanced technologies to track vehicles and capture their number plates, enabling efficient monitoring and enforcement of traffic regulations. By leveraging automatic number plate recognition systems and data analytics, the system facilitates real-time identification of vehicles, aiding in traffic flow optimisation and enhancing security on roadways.


## Installation

For Setting up the Environment, I have used python==3.8 (recommended)

```bash
  git clone https://github.com/zainayaz109/Traffic-Management-System.git
  cd Traffic-Management-System
  conda create --name traffic-management python=3.8
  conda activate traffic-management
  pip install -r requirements.txt
```
## Weight Files
Download weights from
[link](https://drive.google.com/drive/folders/1SMGFrlLxGIQbkTf2si-BDdQFfWTKLyyS?usp=sharing)
and put that in easyocr folder.
## How to use
At first, import the service_obj from main.py

```bash
  from main import service_obj
  import cv2

  img = cv2.imread(path)
  text = service_obj.run(img)
```
    
## Authors

- [Zain Ayaz](https://sites.google.com/view/zainayaz)

