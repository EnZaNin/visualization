from typing import Optional, Any, Dict, List
from pydantic import BaseModel


class LinePlot(BaseModel):
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    size: Optional[str] = None
    style: Optional[str] = None
    sizes: Optional[str] = None
    sort: Optional[bool] = True


class BoxPlot(BaseModel):
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    orient: Optional[str] = None


# class Filter(BaseModel):
#     filters: Optional[List[Dict[str, str]]] = None


class HistPlot(BaseModel):
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    stat: Optional[str] = 'count'
    bins: Optional[Any] = 'auto'
    kde: Optional[bool] = False


class ScatterPlot(BaseModel):
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    style: Optional[str] = None
    size: Optional[str] = None
    palette: Optional[str] = None
    hue_order: Any
    hue_norm: Any
    sizes: Any
    size_order: Any
    size_norm: Any
    markers: Any = True


class JointPlot(BaseModel):
    x: Optional[str] = None
    y: Optional[str] = None
    kind: Optional[str] = 'scatter'
    dropna: Optional[bool] = False
    hue: Optional[str] = None


class CatPlot(BaseModel):
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    row: Optional[str] = None
    col: Optional[str] = None
    col_wrap: Optional[int] = None
    kind: Optional[str] = 'strip'
    height: Optional[int] = 5
    aspect: Optional[int] = 1
    orient: Optional[str] = None
    legend: Optional[bool] = True
    legend_out: Optional[bool] = True
    sharex: Optional[bool] = True
    sharey: Optional[bool] = True


class RelPlot(BaseModel):
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    style: Optional[str] = None
    row: Optional[str] = None
    col: Optional[str] = None
    col_wrap: Optional[int] = None
    row_order: Optional[List[str]] = None
    col_order: Optional[List[str]] = None
    kind = 'scatter'
    height: Optional[int] = 5
    aspect: Optional[int] = 1


class PairPlot(BaseModel):
    hue: Optional[str] = None
    kind: Optional[str] = 'scatter'
    diag_kind: Optional[str] = 'auto'
