A condition monitoring system is developed to test different conditions of a water pump. The model is trained using Support Vector Machine(SVC), Gaussian Naive Bayes(GNB) and an accuracy of about 91 and 92 percentages are obtained respectively.
This dataset is collected under 4 different conditions such as 'Unbalanced', 'Misalignment', 'Normal' and 'Bearing Fault' and is trained.

### Prerequisites:
- Streamlit
- scikit learn
- numpy
- pandas

### Algorithm
1. As the sensor reads the data, it is updated in a csv file.
2. The above csv file can be uploaded in streamlit website and the condition can be tested
3. A more detailed information about the fault can also be known
4. As the model runs, the updated condition along with the timestamp is saved in a csv file. Here, the csv file is 'fault.csv'
5. From fault.csv the database is maintained and analysed to analyse the period of fault occurance
6. We can also visualize the faults encountered over a period of time as selected by the user

### To run the model
Python streamlit.py
