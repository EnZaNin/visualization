from pydantic import BaseModel
from typing import Optional
from datetime import date

class OwidData(BaseModel):

    continent: Optional[str] = None
    location: Optional[str] = None
    date: Optional[date] = None
    # new_cases: Optional[int] = None
    # total_cases: Optional[int] = None
    # new_deaths: Optional[int] = None
    # total_deaths: Optional[int] = None
    # new_tests: Optional[int] = None
    # total_tests: Optional[int] = None
    # new_vaccinations: Optional[int] = None
    # total_vaccinations: Optional[int] = None


    # class Config:
    #     orm_mode = True