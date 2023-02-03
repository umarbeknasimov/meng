from dataclasses import dataclass
import os

from foundations import desc, paths
from averaging.desc import AveragingDesc

@dataclass
class ReportingDesc(desc.Desc):
    averaging_descs: list[AveragingDesc]
