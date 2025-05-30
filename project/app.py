from fastapi import FastAPI                         # Import FastAPI class to create API app
from fastapi.middleware.cors import CORSMiddleware  # Import middleware for handling CORS
from pydantic import BaseModel                       # Import BaseModel for request/response validation
from typing import List                              # Import List for type hinting
import torch                                        # Import PyTorch for model handling

from inference import load_model, predict           # Import functions to load model and make predictions
from embedder import embed_texts                     # Import function to generate text embeddings

app = FastAPI(                                      # Initialize FastAPI app with metadata
    title="NLP Inference API",                      # API title
    description="Automatically generates embeddings and predicts class from input text.",  # API description
    version="1.0.0"                                 # API version
)

app.add_middleware(                                 # Add CORS middleware to allow cross-origin requests
    CORSMiddleware,
    allow_origins=["*"],                            # Allow requests from any origin
    allow_credentials=True,                         # Allow credentials such as cookies
    allow_methods=["*"],                            # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],                            # Allow all headers
)

# Label mapping from integer indices to human-readable topic names
label_mapping = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
model = load_model(model_path='best_model.pth')                       # Load pretrained model from file

class TextInput(BaseModel):                     # Define input data model with Pydantic
    texts: List[str]                            # Input is a list of strings (texts)

class FullOutput(BaseModel):                    # Define output data model with Pydantic
    predictions: List[str]                      # List of predicted class labels as strings
    embeddings: List[List[float]]               # List of embeddings, each is a list of floats

@app.post("/process", response_model=FullOutput, tags=["Unified Process"])  # Define POST endpoint with response model
def process_texts(input_data: TextInput):                   # Endpoint function accepting input_data of type TextInput
    """
    Generate embeddings and predict class for input text.
    """
    embeddings = embed_texts(input_data.texts)              # Generate embeddings for input texts
    predictions_int = predict(model, embeddings, device)    # Predict integer class labels from embeddings using model

    # Convert integer predictions to their string labels using label_mapping dictionary
    predictions_str = [label_mapping.get(int(pred), "Unknown") for pred in predictions_int]

    return {"predictions": predictions_str, "embeddings": embeddings}  # Return predictions and embeddings as JSON
