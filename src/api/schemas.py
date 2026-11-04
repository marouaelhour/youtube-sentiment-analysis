from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    comments: List[str]