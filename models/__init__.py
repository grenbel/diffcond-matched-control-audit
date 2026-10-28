from .build import build_model_from_cfg
import models.TopNet
import models.PoinTr
import models.GRNet
import models.PCN
import models.FoldingNet
import models.SnowFlakeNet
import models.AdaPoinTr
import models.AdaPoinTr_DiffCond
# Cross-backbone validation: UpTrans (ODGNet AAAI 2024) + its DiffCond port
import models.UpTrans
import models.UpTrans_DiffCond