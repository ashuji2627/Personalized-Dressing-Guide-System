# Personalized Dressing Guide System

The **Personalized Dressing Guide System** is a machine learning-powered web application that recommends clothing options based on user preferences and real-world conditions such as weather or occasion. Users can upload their fashion choices or select preferences, and the system uses a trained model to generate suitable outfit suggestions.

Built using **Python**, **TensorFlow**, **ResNet50**, **Google Cloud**, and **Streamlit**, this intelligent fashion assistant provides personalized style recommendations with a clean and responsive interface.

---

##  Features

- **User Preference Upload**: Upload images or specify styles, weather, and occasions to personalize recommendations.
- **Deep Learning-Based Prediction**: Uses a TensorFlow-based model (ResNet50) to classify clothing types and provide intelligent suggestions.
- **Weather & Context Awareness**: Suggestions are influenced by temperature, weather, and occasion for context-sensitive fashion.
- **Real-Time Recommendations**: Outputs results instantly using an optimized model.
- **Streamlit Web Interface**: A lightweight and responsive frontend for users to interact with the system.

---

##  Machine Learning Approach

- **Framework**: TensorFlow
- **Model**: Pre-trained **ResNet50**, fine-tuned on a clothing dataset
- **Input**: Images uploaded by users or text-based clothing preference selection
- **Output**: Predicted clothing category and personalized outfit suggestions based on contextual rules (e.g., weather conditions)

---

##  Technologies Used

- **Python**: Core backend and ML logic
- **TensorFlow**: Deep learning model training and inference
- **ResNet50**: CNN architecture used for image classification
- **Streamlit**: Interactive web UI for input and output display
- **Google Cloud Storage**: For scalable file storage and model hosting
- **OpenWeatherMap API** *(optional)*: To fetch live weather data for dynamic suggestions

---

