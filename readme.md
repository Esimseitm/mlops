docker compose up -d

```
import os
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password123'



```
from roboflow import Roboflow
rf = Roboflow(api_key='ТВОЙ_API_KEY')
project = rf.workspace('moozhil').project('self-driving-car-moozh')
version = project.version(3)
dataset = version.download('yolov8')
```