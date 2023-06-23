Step 1: For Setting up the Environment, I have used python==3.8 (recommended)

-------------------------------------------------------------------
git clone https://github.com/zainayaz109/Traffic-Management-System.git
cd Traffic-Management-System
pip install -r requirements.txt
-------------------------------------------------------------------

Step 2: After that, you need to import the service_obj from main.py

-------------------------------------------------------------------
from main import service_obj

img = cv2.imread(path)
text = service_obj.run(img)
-------------------------------------------------------------------