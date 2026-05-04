import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ref = pd.DataFrame({'avg_confidence': [0.8, 0.9, 0.7], 'num_detections': [5, 6, 4]})
cur = pd.DataFrame({'avg_confidence': [0.4, 0.5, 0.3], 'num_detections': [2, 3, 1]})
r = Report([DataDriftPreset()])
result = r.run(reference_data=ref, current_data=cur)
print(type(result))
print(dir(result))
