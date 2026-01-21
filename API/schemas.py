from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    class_name: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of predicted class")
