# # import torch
# # import torch.nn.functional as F
# # from transformers import AutoModelForSequenceClassification, AutoTokenizer

# # def predict_phishing(text, model_path="./my-phishing-detector"):
# #     # Load model and tokenizer
# #     model = AutoModelForSequenceClassification.from_pretrained(model_path)
# #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    
# #     # Tokenize input
# #     inputs = tokenizer(text, 
# #                        return_tensors="pt", 
# #                        truncation=True, 
# #                        padding="max_length", 
# #                        max_length=128)
    
# #     # Get model prediction
# #     with torch.no_grad():  # Disable gradient calculation for inference
# #         outputs = model(**inputs)
# #         logits = outputs.logits
    
# #     # Convert to probabilities
# #     probabilities = F.softmax(logits, dim=1)
# #     predicted_class = torch.argmax(logits, dim=1).item()
# #     confidence = torch.max(probabilities, dim=1).values.item()
    
# #     # Map to class names (adjust based on your training labels)
# #     class_labels = ["Phishing", "Legitimate"]
    
# #     return {
        
# #         "prediction": class_labels[predicted_class],
# #         "confidence": confidence,
# #         "probabilities": {
# #             "phishing": probabilities[0][0].item(),
# #             "legitimate": probabilities[0][1].item()
# #         }
# #     }

# # # Example usage


# # for sample in test_emails:
# #     result = predict_phishing(sample)
# #     print(result)
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import torch
# import torch.nn.functional as F
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import uvicorn

# # Initialize FastAPI app
# app = FastAPI(title="Phishing Detection API")

# # Load model and tokenizer (global variables)
# MODEL_PATH = "./my-phishing-detector"
# model = None
# tokenizer = None

# @app.on_event("startup")
# async def load_model():
#     global model, tokenizer
#     print("Loading phishing detection model...")
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     print("Model loaded successfully!")

# # Request model
# class EmailText(BaseModel):
#     text: str

# # Response model
# class PhishingResult(BaseModel):
#     is_phishing: bool
#     prediction: str
#     confidence: float
#     phishing_probability: float
#     legitimate_probability: float

# @app.post("/check-phishing", response_model=PhishingResult)
# async def check_phishing(email: EmailText):
#     """
#     Check if an email is phishing or legitimate

#     Send POST request with: {"text": "your email content here"}
#     """
#     if model is None or tokenizer is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     # Tokenize input
#     inputs = tokenizer(email.text, 
#                        return_tensors="pt", 
#                        truncation=True, 
#                        padding="max_length", 
#                        max_length=128)

#     # Get prediction
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits

#     # Convert to probabilities
#     probabilities = F.softmax(logits, dim=1)
#     predicted_class = torch.argmax(logits, dim=1).item()
#     confidence = torch.max(probabilities, dim=1).values.item()

#     # Get probabilities for each class
#     phishing_prob = probabilities[0][0].item()
#     legitimate_prob = probabilities[0][1].item()

#     # Determine if it's phishing (class 0 = Phishing, class 1 = Legitimate)
#     is_phishing = predicted_class == 0
#     prediction = "Phishing" if is_phishing else "Legitimate"

#     return PhishingResult(
#         is_phishing=is_phishing,
#         prediction=prediction,
#         confidence=confidence,
#         phishing_probability=phishing_prob,
#         legitimate_probability=legitimate_prob
#     )

# @app.get("/")
# async def root():
#     return {
#         "message": "Phishing Detection API is running!", 
#         "endpoint": "/check-phishing",
#         "method": "POST",
#         "example": {"text": "Your email content here"}
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn
import os
from pathlib import Path

app = FastAPI(title="Phishing Detection API")

# Use container path - will be mounted from host
MODEL_PATH = "D:/deep_learning/pytorch/PhisingMail/my-phishing-detector"
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("Loading phishing detection model...")

    model_path = os.path.abspath(MODEL_PATH)
    print(f"Model path: {model_path}")

    if not os.path.exists(model_path):
        print(f"ERROR: Model directory not found: {model_path}")
        print("Please ensure your trained model is mounted at /app/my-phishing-detector")
        raise Exception(f"Model directory not found: {model_path}")

    config_file = os.path.join(model_path, "config.json")

    if not os.path.exists(config_file):
        print(f"ERROR: config.json not found in {model_path}")
        print("Available files:")
        try:
            for f in os.listdir(model_path):
                print(f"  - {f}")
        except:
            print("  Could not list directory contents")
        raise Exception("Model config file missing")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Make sure you have saved your model using:")
        print("model.save_pretrained('./my-phishing-detector')")
        print("tokenizer.save_pretrained('./my-phishing-detector')")
        raise e

class EmailText(BaseModel):
    text: str

class PhishingResult(BaseModel):
    is_phishing: bool
    prediction: str
    confidence: float
    phishing_probability: float
    legitimate_probability: float

@app.post("/check-phishing", response_model=PhishingResult)
async def check_phishing(email: EmailText):
    """
    Check if an email is phishing or legitimate

    Send POST request with: {"text": "your email content here"}
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = tokenizer(email.text, 
                       return_tensors="pt", 
                       truncation=True, 
                       padding="max_length", 
                       max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = torch.max(probabilities, dim=1).values.item()

    phishing_prob = probabilities[0][0].item()
    legitimate_prob = probabilities[0][1].item()

    is_phishing = predicted_class == 0
    prediction = "Phishing" if is_phishing else "Legitimate"

    return PhishingResult(
        is_phishing=is_phishing,
        prediction=prediction,
        confidence=confidence,
        phishing_probability=phishing_prob,
        legitimate_probability=legitimate_prob
    )

@app.get("/")
async def root():
    return {
        "message": "Phishing Detection API is running!", 
        "endpoint": "/check-phishing",
        "method": "POST",
        "example": {"text": "Your email content here"}
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None and tokenizer is not None else "unhealthy",
        "model_loaded": model is not None and tokenizer is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# # test_emails = [
# #     # 1. Legitimate internal email
# #     "Hey team, please review the quarterly budget report and send me your comments by Friday. We need to ensure all line items are reconciled before submitting to finance.",

# #     # 2. Legitimate client update
# #     "Dear Client, as discussed in our last call, we have completed the setup of your cloud environment. You can access the resources using your credentials. Let us know if you have any questions.",

# #     # 3. Legitimate invoice confirmation
# #     "Hello, your payment for Invoice #1122 has been successfully processed. Thank you for your prompt payment. No further action is required.",

# #     # 4. Legitimate project collaboration email
# #     "Hi John, can you send me the latest data files for the wind energy simulation project? I want to integrate them into the main model by end of day.",

# #     # 5. Legitimate newsletter / update
# #     "Dear subscriber, here is your weekly update from the Tech Insights newsletter. This week we cover cloud cost optimization strategies and upcoming webinars.",

# #     # 6. Phishing / fake prize scam
# #     "Congratulations! You've been selected as a winner of a $1,000,000 cash prize. Click the link below to claim your reward immediately: www.fake-prize-link.com",

# #     # 7. Phishing / bank spoof
# #     "Dear Customer, we've noticed unusual activity on your account. Please verify your login here immediately to avoid account suspension: https://securebank.fake-login.com",

# #     # 8. Phishing / malicious attachment
# #     "Hello, please see the attached invoice for your recent order. Open the document and follow instructions to process payment. Attachment: invoice_2025.docm",

# #     # 9. Phishing / urgent CEO request
# #     "Hi, I need you to process an urgent wire transfer of $50,000 to the account below. Confirm once done. Account Name: Royal Importers Ltd. Account Number: 9988776655",

# #     # 10. Legitimate team discussion
# #     "Hey guys, I would like to discuss the recent changes in our risk management strategy for the wind energy portfolio. Please share your availability for a call next week.",

# #     # 11. Legitimate support response
# #     "Hi Priyanshu, your support ticket has been resolved. Please try logging in again and let us know if the issue persists.",

# #     # 12. Phishing / delivery scam
# #     "We attempted delivery of your package but could not leave it. Please verify your address immediately at: http://shipfast-delivery.fake/confirm Tracking: SF-771-9932"
# # ]