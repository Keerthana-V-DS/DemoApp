import pickle
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
 
# loading the trained model
pickle1_in = open('titanic_spaceship_classifier2.pkl', 'rb') 
classifier = pickle.load(pickle1_in)

pickle2_in = open('titanic_spaceship_encoder1.pkl', 'rb') 
scaler = pickle.load(pickle2_in)

 
@st.cache()
  

    
# defining the function which will make the prediction using the data which the user inputs 
def prediction(HomePlanet, CryoSleep, Destination, VIP, Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck):   
    # Pre-processing user input 
        
    ## Age
    if Age < 14:
        Age = 'Kids Under 14'
    elif (Age > 14) and (Age < 35):
        Age = '14-35'
    elif (Age > 35) and (Age < 50):
        Age = '35-50'
    else:
        Age = 'Older than 50'
    
    my_list = [HomePlanet,Destination,Age]
    arr = np.array(my_list)
    transformed_data = scaler.fit_transform(arr.reshape(1, -1)).toarray()
    test = [CryoSleep, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck]
    test.extend(transformed_data)
    test = np.array(test)
    
    
    test = [CryoSleep, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck]
    test.extend(transformed_data)

    # Making predictions 
    prediction = classifier.predict(test.reshape(1, -1))
     
    if prediction == 0:
        pred = 'Not Transported'
    else:
        pred = 'Transported'
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Titanic Spaceship Transportation Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    st.text("")
    image = Image.open('spaceship.jpg')

    st.image(image, caption='Sunrise by the mountains')
    st.text("")
    # following lines create boxes in which user can enter data required to make prediction 
    HomePlanet = st.selectbox('HomePlanet',("Earth","Mars","Europa",'Others'))
    Destination = st.selectbox('Destination',('TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e', 'Others'))
    CryoSleep = st.selectbox('CryoSleep Status',("Yes","No"))
    VIP = st.selectbox('Is a VIP?', ('Yes','No'))
    
    Age = st.number_input("Passenger's age", value = 20, step = 1) 
    RoomService = st.number_input("Money spent on room service")
    FoodCourt = st.number_input("Money spent on FoodCourt")
    ShoppingMall = st.number_input("Money spent on Shopping mall")
    Spa = st.number_input("Money spent on spa")
    VRDeck = st.number_input("Money spent on VRDeck")
    
    passenger_name  = st.text_input("What's the passenger's name?", value = 'Name')
    
    st.file_uploader('Upload your doc here.', accept_multiple_files=False)
    
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(HomePlanet, CryoSleep, Destination, VIP, Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck) 
        st.success('The passenger is {}'.format(result))
        print(HomePlanet,'to',Destination)
     
if __name__=='__main__': 
    main()
