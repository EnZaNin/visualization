from pydantic import BaseModel

class OwidData(BaseModel):

    id: int
    location: str
    new_cases: str

    class Config:
        orm_mode = True