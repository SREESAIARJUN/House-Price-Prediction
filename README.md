# **🏡 House Price Prediction API & Web App**  
🚀 **FastAPI-based Machine Learning API** + **Streamlit Web App** for predicting house prices using ONNX models.  

---

## **📌 Project Overview**  
This project provides:  
✅ **A FastAPI backend** for predicting house prices using a machine learning model.  
✅ **An ONNX-optimized model** for memory-efficient inference.  
✅ **A Streamlit web app** to interactively predict house prices.  
✅ **Deployment on Render & Streamlit Cloud** for real-time predictions.  

---

## **🛠️ Tech Stack**  
- **Backend:** FastAPI  
- **Model Format:** ONNX (Converted from XGBoost)  
- **Frontend:** Streamlit  
- **Deployment:** Render (API) + Streamlit Cloud (Web App)  
- **Additional Tools:** Pandas, NumPy, ONNX Runtime, Requests  

---

## **🚀 FastAPI Setup & Usage**  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/House-Price-Prediction.git
cd House-Price-Prediction
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the FastAPI Server**  
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
FastAPI will be available at:  
**➡️ `http://localhost:8000`**  
Swagger UI: **`http://localhost:8000/docs`**  

### **4️⃣ Test the API with cURL**  
```bash
curl -X 'POST' \
  'https://house-price-prediction-oo91.onrender.com/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "MedInc": 3.5,
  "HouseAge": 15,
  "AveRooms": 5.4,
  "AveBedrms": 1.2,
  "Population": 1500,
  "AveOccup": 3.0,
  "Latitude": 37.5,
  "Longitude": -122.0
}'
```

---

## **🎨 Streamlit Web App Setup**  
### **1️⃣ Install Streamlit**  
```bash
pip install streamlit requests
```

### **2️⃣ Run the Web App**  
```bash
streamlit run streamlit_app.py
```

### **3️⃣ Web App Features**  
✅ **Intuitive UI:** Adjust house features using sliders.  
✅ **Real-time Predictions:** Calls the FastAPI backend for instant results.  
✅ **User-friendly Design:** Built with Streamlit for an elegant experience.  

---

## **🔗 Live Deployment**  
- **🔹 FastAPI Backend:** [`house-price-prediction-oo91.onrender.com`](https://house-price-prediction-oo91.onrender.com)  
- **🔹 Streamlit Web App:** [`housepricepredictionbyarjun.streamlit.app`](https://housepricepredictionbyarjun.streamlit.app)  

---

## **📜 License**  
This project is licensed under the **Apache 2.0 License**.   

---

## **💡 Future Improvements**  
🚀 **Optimize model further with ONNX quantization**  
🚀 **Deploy Streamlit app on Streamlit Cloud**  
🚀 **Add authentication to API for secure access**  

---

## **💬 Feedback & Contributions**  
Got ideas or improvements? **Feel free to contribute!**  
📧 Contact: **sreesaiarjunwork.com**  

---
