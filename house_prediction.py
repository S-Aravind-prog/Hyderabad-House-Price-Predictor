import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('pipe.pkl','rb'))
bhk = 3
area = 2500
test_input = pd.DataFrame(
    [[
        'Kondapur',
        'Apartment',
        'Ready to move',
        bhk,
        area,
        area/bhk
    ]],
    columns=[
        'location',
        'PropertyType',
        'building_status',
        'BHK',
        'area_insqft',
        'AreaPerRoom'
    ]
)
prediction = pipe.predict(test_input)

print("Predicted Price (Lakhs):", prediction[0])