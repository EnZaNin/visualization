from typing import Optional
from pydantic import BaseModel

class LinePlot(BaseModel):
    # data_source: str
    # chart_name: str

    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    # size: Optional[str] = None
    # style: Optional[str] = None
    # palette: Optional[str] = None
    # hue_order: Optional[str] = None
    # hue_norm: Optional[str] = None
    # sizes: Optional[str] = None
    # size_order: Optional[str] = None
    # size_norm: Optional[str] = None
    # dashes: Optional[str] = True
    # markers: Optional[str] = None
    # style_order: Optional[str] = None
    # units: Optional[str] = None
    # estimator: Optional[str] = 'mean'
    # ci: Optional[int] = 95
    # n_boot: Optional[int] = 1000
    # seed: Optional[str] = None
    # sort: Optional[str] = True
    # err_style: Optional[str] = 'band'
    # err_kws: Optional[str] = None
    # legend: Optional[str] = 'auto'
    # ax: Optional[str] = None

    class Config:
        orm_mode = True