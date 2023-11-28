#from .dataprocessor import AutosomeDataProcessor
from .first_layers_block import UnlockDNA_FirstLayersBlock
from .coreblock import UnlockDNA_CoreBlock
from .final_layers_block import UnlockDNA_FinalLayersBlock
#from .predictor import AutosomePredictor
#from .trainer import AutosomeTrainer

__all__ = ("UnlockDNA_FirstLayersBlock",
           "UnlockDNA_CoreBlock",
           "UnlockDNA_FinalLayersBlock")

#__all__ = ("AutosomeDataProcessor",
#           "UnlockDNA_FirstLayersBlock",
#           "UnlockDNA_CoreBlock",
#           "UnlockDNA_FinalLayersBlock",
#           "AutosomePredictor",
#           "AutosomeTrainer")
