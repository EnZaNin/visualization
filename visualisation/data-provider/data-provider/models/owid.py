from sqlalchemy import Column, Integer, String

from ..database import Base

class Owid(Base):
    __tablename__ = 'owid'

    location = Column(String, primary_key=True)
    date = Column(Integer, primary_key=True)