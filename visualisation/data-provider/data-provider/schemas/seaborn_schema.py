from typing import Optional
from pydantic import BaseModel

class LinePlot(BaseModel):
    # data_source: str
    # chart_name: str

    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None

    class Config:
        orm_mode = True