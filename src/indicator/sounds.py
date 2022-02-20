# A file to contain functions used to indicate a step has completed
# Author: Nathan Fragar

def beep_when_done():
    
    """When processing has completed for a cell, perform a sound indicating beep

    Returns
    -------
    """
    
    ## Import up sound alert dependencies
    from IPython.display import Audio, display
    
    display(Audio(url='https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg', autoplay=True))
    
    return