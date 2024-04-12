import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, date
import csv   


def extract_features(fft_values):
    """
    Extracts statistical features from FFT values.
    Args:
        fft_values (numpy.ndarray): 1D array containing FFT values.
    Returns:
        dict: Dictionary containing the extracted features.
    """
    # Calculate RMS (Root Mean Square)
    rms = np.sqrt(np.mean(fft_values**2))

    # Calculate Crest Factor
    crest_factor = np.max(np.abs(fft_values)) / rms

    # Calculate Kurtosis
    kurt = kurtosis(fft_values)

    # Calculate Skewness
    skew_value = skew(fft_values)

    # Calculate Shape Factor
    shape_factor = rms / (np.mean(np.abs(fft_values)))

    # Calculate PeakPeak value
    peak_to_peak = np.max(fft_values) - np.min(fft_values)

    features = {
        'rms': rms,
        'crest_factor': crest_factor,
        'kurtosis': kurt,
        'skewness': skew_value,
        'shape_factor': shape_factor,
        'peak_to_peak': peak_to_peak
    }

    return features

def predictions(pickled_model,features_y):
    input_data_as_numpy_array = np.asarray(features_y)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = pickled_model.predict(input_data_reshaped)
    time = datetime.now()

    if (prediction[0] == 'unbalance'):
        st.title("Fault detected in water pump: Unbalanced fault")
        col1, col2,col3,col4= st.columns([1,1,1,1])
        with col1:
            b1 = st.button("More about fault")
        with col2:
            b2 = st.button("How to rectify?")
        if b1:
            st.write("Know about Unbalance Fault")
            st.image("unbalance.jpeg")
            st.write("An unbalance fault in water pumps occurs when the distribution of mass within the pump impeller or shaft is uneven, causing vibrations during operation. These vibrations can lead to reduced efficiency, increased wear and tear, and potential damage to the pump and associated equipment.")
        if b2:
            st.write("Major reasons for the occurace of Unbalance fault")
            st.image("unbalance.jpeg")
            st.write("Inspection: Conduct a thorough inspection of the pump and associated components to identify the source of the unbalance. This may involve visual inspection, measurement of vibration levels, and assessment of the impeller and shaft for signs of damage or irregularities.")
            st.write("Dynamic Balancing: If the unbalance is due to uneven mass distribution in the impeller or shaft, dynamic balancing may be necessary. This process involves adding or removing weight from specific locations to ensure that the rotating components are evenly balanced. Specialized equipment such as balancing machines may be required for this procedure.")
            st.write("Repair or Replacement: Depending on the severity of the unbalance fault and the condition of the pump components, repair or replacement may be necessary. Damaged or worn parts should be repaired or replaced to restore the pump to proper working condition.")
            st.write("Alignment: Proper alignment of the pump shaft and motor shaft is crucial for minimizing vibration and ensuring efficient operation. Ensure that the pump shaft and motor shaft are aligned correctly according to manufacturer specifications.")
            st.write("Regular Maintenance: Implement a regular maintenance schedule to monitor the condition of the pump and perform preventative maintenance tasks. This may include checking alignment, lubricating bearings, and inspecting for signs of wear or damage.")
            st.write("Monitoring: Install vibration monitoring systems to continuously monitor the vibration levels of the pump during operation. This allows for early detection of any issues that may indicate an unbalance fault or other mechanical problems.")
            st.write("By following these steps, the unbalance fault in water pumps can be identified and rectified, ensuring smooth and efficient operation while minimizing the risk of damage and downtime.")

    elif (prediction[0] == 'misalignment'):
        st.title("Fault detected in water pump: Misalignment fault")
        col1, col2,col3,col4= st.columns([1,1,1,1])
        with col1:
            b1 = st.button("More about fault")
        with col2:
            b2 = st.button("How to rectify?")
        if b1:
            st.write("Know about misalignment Fault")
            st.image("misalignment.jpeg")
            st.write("Misalignment faults in machinery occur when components like shafts aren't properly aligned. This leads to increased vibration, reduced efficiency, and premature wear. Detecting and rectifying misalignment promptly is crucial to prevent further damage and maintain optimal performance. Using precision tools for measurement, adjusting positions, and regular maintenance are key steps in addressing misalignment. By ensuring correct alignment, machinery operates smoothly, minimizing downtime and extending its lifespan.")
        if b2:
            st.write("Major reasons for the occurace of misalignment fault")
            st.image("misalignment.jpeg")
            st.write("Measurement: Use precision alignment tools such as dial indicators or laser alignment systems to accurately measure the misalignment between the pump shaft and motor shaft. This helps determine the extent and direction of misalignment.Adjustment: Adjust the position of the pump or motor to correct the misalignment. This may involve shimming, repositioning mounting bolts, or adjusting motor mounts to achieve proper alignment.Alignment Checks: After making adjustments, recheck the alignment to ensure that it meets the manufacturer's specifications. Repeat the adjustment process if necessary until the alignment is within acceptable tolerances.")
            st.write("Bearing Inspection: Inspect the pump bearings for signs of damage or wear caused by the misalignment. Replace any damaged bearings to prevent further issues.")
            st.write("Coupling Inspection: Examine the coupling connecting the pump shaft and motor shaft for signs of damage or misalignment. If the coupling is worn or damaged, replace it with a new one to ensure proper alignment and transmission of power.")
            st.write("Tightening and Securing: Ensure that all mounting bolts, fasteners, and coupling elements are properly tightened and secured to maintain alignment under operational conditions.")
            st.write("Regular Maintenance: Implement a regular maintenance schedule to monitor alignment and perform adjustments as needed. Regular checks can help identify and address misalignment issues before they cause significant damage or downtime.")
            st.write("Training: Provide training to personnel involved in the installation and maintenance of water pumps to ensure proper alignment procedures are followed consistently.")  
    elif (prediction[0] == 'bearing'):
        st.title("Fault detected in water pump: Bearing fault")
        col1, col2,col3,col4= st.columns([1,1,1,1])
        with col1:
            b1 = st.button("More about fault")
        with col2:
            b2 = st.button("How to rectify?")
        if b1:
            st.write("Know about Bearing Fault")
            st.image("bearing.jpeg")
            st.write("Bearing faults in machinery occur when bearings degrade due to factors like wear, lubrication issues, or contamination. Symptoms include increased noise, vibration, and heat. Prompt detection and rectification are essential to prevent further damage to equipment. Techniques such as vibration analysis and lubrication checks aid in identifying and addressing bearing faults. Regular maintenance, including bearing replacement and proper lubrication, helps maintain smooth operation. Addressing bearing faults promptly enhances equipment reliability, prolongs lifespan, and minimizes downtime")
        if b2:
            st.write("Major reasons for the occurace of bearing fault")
            st.image("bearing.jpeg")
            st.write("Rectifying bearing faults is essential for maintaining machinery efficiency and preventing costly downtime. The steps to rectify bearing faults typically involve:")
            st.write("Identification: Diagnose the specific bearing fault through noise analysis, vibration monitoring, or visual inspection.")
            st.write("Replacement: If the bearing is damaged or worn, replace it with a new one of the correct size and specification.")
            st.write("Lubrication: Ensure proper lubrication to reduce friction and wear on the bearing surfaces.")
            st.write("Alignment: Check and adjust shaft alignment to prevent undue stress on the bearing.")
            st.write("Tightening: Ensure proper tightening of mounting bolts to prevent bearing misalignment.")
            st.write("Monitoring: Implement regular monitoring to detect any early signs of bearing wear or failure.")
            st.write("By following these steps, bearing faults can be rectified effectively, ensuring smooth machinery operation and prolonging equipment lifespan.")  
    else:
        st.title("Normal")
        st.write("Hurray! Your machinery is perfect")
    fields=[prediction[0],time]
    with open(r'fault.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def process_data(data):
    # Extract the acceleration data for axis_y
    y_data = data['axis_y']

    # Check if the data is a list or series of dictionaries
    if isinstance(y_data.iloc[0], dict):
        # Define the keys for the numerical features you want to extract
        feature_keys = ['rms', 'crest_factor', 'kurtosis', 'skewness', 'shape_factor', 'peak_to_peak']

        # Extract the numerical values from the dictionaries
        numerical_values = [[row[key] for key in feature_keys] for row in y_data]

        # Convert the list of numerical values to a NumPy array or pandas DataFrame
        y = np.array(numerical_values)
        # print(type(y))
        # y = pd.DataFrame(numerical_values, columns=feature_keys)
    else:
        y = y_data.values
        # print(type(y))

    x_data = data['axis_x']
    x = x_data.values
    z_data = data['axis_z']
    z = z_data.values

    # Perform FFT on axis_y
    fft_y = np.fft.fft(y)
    fft_x = np.fft.fft(x)
    fft_z = np.fft.fft(z)

    # Extract features for axis_y
    features_y = extract_features(np.abs(fft_y))
    print(type(features_y))

    for keys,values in features_y.items():
        if keys=="rms":
            rms = values
        elif keys == "crest_factor":
            crest_factor = values
        elif keys == "kurtosis":
            kurtosis = values
        elif keys == "skewness":
            skewness = values
        elif keys == "shape_factor":
            shape_factor = values
        else :
            peak_to_peak = values

    return fft_x,fft_z,fft_y,rms,crest_factor,kurtosis,skewness,shape_factor,peak_to_peak

def app():
    st.title("Feature Extraction and Prediction")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
        # Process the data
        features_y = []
        fft_x,fft_z,fft_y,rms,crest_factor,kurtosis,skewness,shape_factor,peak_to_peak = process_data(data)
        features_y.append(rms)
        features_y.append(crest_factor)
        features_y.append(kurtosis)
        features_y.append(skewness)
        features_y.append(shape_factor)
        features_y.append(peak_to_peak)

        col1, col2,col3= st.columns([1,1,1])
        with col1:
            vis_x = st.button("Visualize X_axis")
        if vis_x:
            fig1, ax = plt.subplots(figsize=(10, 6))
            ax.plot(np.abs(fft_x))
            ax.set_title("Frequency Spectrum of axis x")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig1)
        with col2:
            vis_y = st.button("Visualize y_axis")
        if vis_y:
            fig2, ay = plt.subplots(figsize=(10, 6))
            ay.plot(np.abs(fft_y))
            ay.set_title("Frequency Spectrum of axis y")
            ay.set_xlabel("Frequency")
            ay.set_ylabel("Amplitude")
            st.pyplot(fig2)
        with col3:
            vis_z = st.button("Visualize z_axis")
        if vis_z:
            fig3, az = plt.subplots(figsize=(10, 6))
            az.plot(np.abs(fft_z))
            az.set_title("Frequency Spectrum of axis z")
            az.set_xlabel("Frequency")
            az.set_ylabel("Amplitude")
            st.pyplot(fig3)

        # Get model selection
        model_choice = st.selectbox("Select a model", ["--","SVC", "GNB"])

        # Make prediction using the selected model
        # (You'll need to implement this part based on your model)
        if model_choice == "SVC":
            pickled_model = pickle.load(open('svc 1.pkl', 'rb'))
            predictions(pickled_model,features_y)
        elif model_choice == "GNB":
            pickled_model = pickle.load(open('gnb 1.pkl', 'rb'))
            predictions(pickled_model,features_y)
        else:
            st.text("Select a model to predict the fault")


# Sidebar navigation
selected_feature = st.sidebar.selectbox("Select Feature", ["Condition Monitoring","Feature Prediction","My data"])

if selected_feature == "Feature Prediction":
    app()
if selected_feature == "Condition Monitoring":
    st.title("Condition Monitoring")
    st.write("Condition monitoring (CM) is a maintenance approach that predicts machine health and safety through the combination of machine sensor data that measures vibration and other parameters (in real-time) with state-of-the-art machine monitoring software. This approach enables plant maintenance technicians to remotely monitor the health of each individual piece of machinery and also offers a holistic, plant-wide view of mechanical operations.  Condition monitoring software sends an alert whenever a change is detected in machine health, enabling your maintenance technicians to immediately assess the situation and determine if corrective action is required.")
    st.image("cm.jpeg")
    col1, col2,col3,col4= st.columns([1,1,1,1])
    with col1:
        abt_h = st.button("About Hardware")
    with col2:
        abt_s = st.button("About Software")
    if abt_h:
        st.title("Hardware")
        st.write("Sensor: Accelerometer")
        st.write("Controller: Arduino AtMega")
    if abt_s:
        st.title("Software")
        st.write("Machine learning (ML) is a branch of artificial intelligence (AI) and computer science that focuses on the using data and algorithms to enable AI to imitate the way that humans learn, gradually improving its accuracy.")
        st.write("ML Algorithms used")
        st.write("1. Support Vector Machine")
        st.write("2. Gaussian Naive Bayes")
        st.write("3. K Nearest Neighbours")
    st.title("Benefits of Condition Monitoring")
    st.image("benefit.jpg")
    st.write("The proactive nature of condition monitoring is an innovative step forward on several levels for some manufacturers. First, plant personnel are safer and thus, we are all collectively safer. Second, plant managers can prevent unplanned downtime due to machine failure while simultaneously making the most of planned maintenance downtime by servicing multiple machines and addressing all known problems at the same time. Further, condition monitoring also eliminates unnecessary—and wasted—costs associated with over maintaining healthy machines based on the static metric of operating hours alone.")
    st.write("Although condition monitoring is a tried and true industrial maintenance tool, it is only just beginning to be leveraged effectively in a wider array of manufacturing industries. Today’s condition monitoring systems can do much more for us—financially, operationally, and most importantly, from a safety perspective. Today’s condition monitoring solutions are highly reliable and have been proven extremely effective across multiple manufacturing industries. Thus, for manufacturers who adopt condition based maintenance techniques, the risk is low and the reward is high")
    st.title("Advantages of proactive/predictive maintenance")
    st.image("advantage.jpg")
    st.write("The heart and soul of industrial businesses around the world are our manufacturing facilities. Adopting proactive predictive maintenance techniques is much more than just good plant management, it is good business. Today, only (3 to 5%) of available data is being used to make important operational decisions. Digitalization is key to unlocking the vast amounts of untapped information embedded in your operation. Connect data, insights, and self-learning models across your entire operation to increase capital efficiency and profitability, decrease costs, and better allocate valuable resources.")
    st.write("Anticipating machine failures before they occur, allows you to catalyze improvements that create positive ripple effects for the entire enterprise, such as:")
    st.write("Minimize downtime, Maximize production (90%) of failures are NOT time-based.  For many assets, failure can mean a substantial or total loss of production, often worth tens of thousands to millions per day.  Often industries tend to focus on the larger, more expensive machines at the expense of ignoring the smaller supporting machines.  Focusing on the machines that “make the money” is important but so too is focus on those machines without which the money making machine can’t operate.")
    st.write("Increase safety - Relying exclusively on hand-held devices for monitoring machine health can expose factory workers to unnecessary risks in our highly automated factories. Further, occasional catastrophic breakdowns due to maintenance gaps can increase employee exposure to hazardous conditions and potential environmental disasters.")
    st.write("Reduce maintenance costs- When viewed on a per-asset basis, maintenance costs for plant-wide assets can appear modest. However, when viewed collectively across the dozens, hundreds, or even thousands of assets in a typical plant, these costs can be appreciable. Reducing the maintenance costs on each asset through effective condition monitoring—even by a mere 10%—has a large impact on plant profitability. Condition Monitoring is a planning tool that allows more effective insight in planning and asset management, allowing maintenance to be done in advance of a functional failure.")
    st.write("Reduce hidden costs - Direct (traditional) maintenance costs are predictable and manageable. Indirect (hidden) maintenance costs, both stealthy and steep, can accrue to be up to 5X higher. For many plants, reducing these hidden costs is a mandate that requires us to shift from the traditional reactive approach (“fix it when it breaks”) to a proactive, reliability-based approach.")

if selected_feature == "My data":
    st.title("Welcome!")

    # Read the CSV file
    data = pd.read_csv('fault.csv', parse_dates=['time'])

    # Convert the datetime column to datetime format
    data['time'] = pd.to_datetime(data['time'])

    # Get the current date
    today = date.today()

    # Filter the data for today's date
    today_data = data[data['time'].dt.date == today]

    # Check if there are any values other than 'normal' for today
    abnormal_values = today_data[today_data['prediction'] != 'normal']['prediction'].unique()

    if len(abnormal_values) > 0:
        st.title(f"Oops, {len(abnormal_values)} Abnormalities found today !")
    else:
        st.title(f"Hurray! Your Machine worked 'Normal' today")

    col1,col2,col3 = st.columns([1,1,1])
    with col1:
        hist = st.button("View History")
    if hist:
        st.write(data.tail())
    with col2:
        view = st.button("View Summary")
    if view:
        # Sort the data by datetime
        data = data.sort_values('time', ascending=False)

        # Initialize a dictionary to store the results
        results = {}

        # Loop through the rows of the DataFrame
        for index, row in data.iterrows():
            prediction = row['prediction']
            date = row['time']

            # Check if the prediction value is not 'normal'
            if prediction != 'normal':
                # If the prediction value is not in the results dictionary, add it
                if prediction not in results:
                    results[prediction] = {'recent_occurrence': date, 'previous_occurrence': None}
                else:
                    # If the prediction value is already in the dictionary, update the next occurrence date
                    if results[prediction]['previous_occurrence'] is None or date < results[prediction]['recent_occurrence']:
                        results[prediction]['previous_occurrence'] = date

        # Print the results
        st.title("Fault Occurrences")
        for prediction, dates in results.items():
            st.title(prediction)
            st.write(f"Recent occurrence : {dates['recent_occurrence']}")
            st.write(f"Previous occurrence : {dates['previous_occurrence']}")

    with col3:
        visual = st.button("Visualize")
    if visual:
        def count_faults(df):
            normal_count = len(df[df['prediction'] == 'normal'])
            misalignment_count = len(df[df['prediction'] == 'misalignment'])
            bearing_count = len(df[df['prediction'] == 'bearing'])
            unbalance_count = len(df[df['prediction'] == 'unbalance'])
            return normal_count, misalignment_count, bearing_count, unbalance_count

        # Streamlit app
        st.title("Fault Analysis")

        # Date range selection
        start_date = st.date_input("Start Date", value=data['time'].dt.date.min(), min_value=data['time'].dt.date.min(), max_value=data['time'].dt.date.max())
        end_date = st.date_input("End Date", value=data['time'].dt.date.max(), min_value=data['time'].dt.date.min(), max_value=data['time'].dt.date.max())

        # Filter data based on selected date range
        date_range_data = data[(data['time'].dt.date >= start_date) & (data['time'].dt.date <= end_date)]

        # Count faults
        normal_count, misalignment_count, bearing_count, unbalance_count = count_faults(date_range_data)

        labels = []
        sizes = []

        if normal_count>0:
            labels.append('Normal')
            sizes.append(normal_count)
        if misalignment_count>0:
            labels.append('Misalignment')
            sizes.append(misalignment_count)
        if bearing_count>0:
            labels.append('Bearing')
            sizes.append(bearing_count)
        if unbalance_count>0:
            labels.append('Unbalanced')
            sizes.append(unbalance_count)

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
    