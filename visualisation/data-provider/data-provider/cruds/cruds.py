from sqlalchemy.orm import Session
from ..models.owid import Owid
from ..schemas.owid import OwidData

class DataDbTools:
    def __init__(self, db: Session):
        self._db = db
    
    # def create_table(self, row: OwidData) -> Owid:
    #     db_row = Owid(location=row.location, new_cases=row.new_cases)
    #     self._db.add(db_row)
    #     self._db.commit()
    #     self._db.refresh(db_row)
    #     return db_row
    
    def get_from(self):
        return self._db.query(Owid).offset(0).all()
# NALEZY ROZROZNIC JEDEN WYMIAR I WIELEWYMIAROW