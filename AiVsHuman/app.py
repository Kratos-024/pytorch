
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import os
import uvicorn



class AIHumanTextClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer from the local checkpoint"""
        print(f"Loading model from {self.model_path}...")

        if not os.path.exists(self.model_path):
            print(f"❌ Path does not exist: {self.model_path}")
            return False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, 
                local_files_only=True
            )
            self.model.eval()
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def clean_text(self, text):
        """Clean text for preprocessing"""
        text = str(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        return text.strip()

    def predict_single(self, text):
        """Predict a single text sample"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        cleaned_text = self.clean_text(text)

        inputs = self.tokenizer(
            cleaned_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][int(predicted_class)].item()

        prediction = "AI" if predicted_class == 1 else "Human"
        return {
            "prediction": prediction, 
            "confidence": confidence,
            "raw_scores": predictions[0].tolist()
        }

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    human_score: float
    ai_score: float

app = FastAPI(
    title="AI/Human Text Classifier API",
    description="API for classifying text as AI-generated or Human-written",
    version="1.0.0"
)

classifier = None


def initialize_model():
    """Initialize the model from possible paths"""
    global classifier

    possible_paths = [
        "D:/deep_learning/pytorch/AiVsHuman/3500", 
        "D:/deep_learning/pytorch/AiVsHuman/3500",  
        os.path.join(os.getcwd(), "3500"),
        r"D:/deep_learning/pytorch/AiVsHuman/3500"  
    ]

    print("🔍 Current working directory:", os.getcwd())
    print("🔄 Looking for model...")

    for path in possible_paths:
        print(f"   Checking: {path}")
        if os.path.exists(path):
            print(f"   ✅ Found model at: {path}")
            classifier = AIHumanTextClassifier(path)
            if classifier.load_model():
                print(f"   🎉 Model loaded successfully!")
                return True
        else:
            print(f"   ❌ Not found: {path}")

    print("❌ Could not load model. Make sure 'ai_human_model' folder exists in current directory")
    return False


def run_quick_tests():
    """Run some quick tests to verify the model works"""
    if classifier is None or classifier.model is None:
        print("⚠️  Skipping tests - model not loaded")
        return

    test_cases = [
        {
            "text": "Check out this amazing sunset I captured! #nature #photography",
        },
        {
            "text": "Here are 5 key benefits of implementing AI in business operations: improved efficiency, cost reduction, better analytics, automation, and enhanced customer experience.",
        },
        {
            "text": "Just had the best pizza ever! Can't believe how good this place is 😋",
        },
        {
            "text": "Completed DSA Question 1.Two sum 2.palindrone",
        },
        {
            "text": "Bihar Government is not that good.",
        }
    ]

    print("\n🧪 Running Quick Tests:")
    print("="*50)

    correct = 0
    for idx, case in enumerate(test_cases, 1):
        try:
            result = classifier.predict_single(case["text"])
            print(result)
            is_correct = result["prediction"] == case["expected"]
            if is_correct:
                correct += 1

            status = "✅" if is_correct else "❌"
            print(f"Test {idx} | {result['prediction']:5} | {result['confidence']:.1%} | {status}")
        except Exception as e:
            print(f"Test {idx} | ERROR: {e}")

    print("="*50)
    print(f"📊 Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases):.1%})")


@app.on_event("startup")
async def startup_event():
    """Initialize model when API starts"""
    print("🚀 Starting AI/Human Text Classifier API...")
    initialize_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🤖 AI/Human Text Classifier API",
        "status": "Model loaded" if (classifier and classifier.model) else "Model not loaded",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "test": "/test"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = classifier is not None and classifier.model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "message": "Ready to classify text!" if model_loaded else "Model not available"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """Predict if text is AI-generated or Human-written"""
    if classifier is None or classifier.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check if 'ai_human_model' folder exists."
        )

    try:
        result = classifier.predict_single(input_data.text)
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            human_score=result["raw_scores"][0],
            ai_score=result["raw_scores"][1]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/test")
async def run_api_tests():
    """Run built-in test cases"""
    if classifier is None or classifier.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )

    test_texts = [
        "Amazing sunset today! Nature is beautiful 🌅 #photography",
        "The implementation of machine learning algorithms requires careful consideration of data preprocessing, feature selection, and model validation techniques.",
        "Just spilled coffee all over my laptop... Monday mood 😭",
        "Research indicates that artificial intelligence can improve operational efficiency by up to 40% when properly integrated into existing workflows."
    ]

    results = []
    for text in test_texts:
        try:
            result = classifier.predict_single(text)
            results.append({
                "text": text[:60] + "..." if len(text) > 60 else text,
                "prediction": result["prediction"],
                "confidence": round(result["confidence"], 3),
                "human_score": round(result["raw_scores"][0], 3),
                "ai_score": round(result["raw_scores"][1], 3)
            })
        except Exception as e:
            results.append({
                "text": text[:60] + "...",
                "error": str(e)
            })

    return {
        "total_tests": len(test_texts),
        "results": results
    }


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 AI/Human Text Classifier")
    print("=" * 60)

    if initialize_model():
        run_quick_tests()

        print("\n" + "=" * 60)
        print("🌐 Starting Web Server...")
        print("📖 API Docs: http://localhost:8000/docs")
        print("🔍 Health: http://localhost:8000/health")
        print("🧪 Test: http://localhost:8000/test")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 60)
    else:
        print("⚠️  Model not loaded but starting server anyway...")
        print("   Make sure 'ai_human_model' folder is in the same directory as this script")

    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
