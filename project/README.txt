This project predict prices based on orderbook data.

1. Orderbook Construction
Orderbook construction is implemented at "snapshot.cpp"
You can execute the code by "bash process_data.sh". The reusult is saved under the folder, "outputs".

2. Prediction
I have implemented the file, "main.py" to generate fair prices and save it as csv file.
You can execute the code by "python main.py".
The detail analysis as to feature and model selection is explained at "Order Flow Analysis.ipynb".
These analysis utlizes the custom library "libs". Please take a look at it to see more detail implementations.