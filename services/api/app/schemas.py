from pydantic import BaseModel

class QueryInput(BaseModel):
    text: str

class QueryResponse(BaseModel):
    response_text: str