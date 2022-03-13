import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
import os
sayac=0
"""for q in os.listdir("./voices/"):
    for i in range(3):
        ng, fs=  sf.read("./voices/"+str(q)+"/record_"+str(i)+".wav", dtype='float32')
        ng1=ng[:int(ng.shape[0]/2)]
        ng2=ng[int(ng.shape[0]/2):]
        write("./voices/"+str(q)+"/record_"+str(i)+".wav", 44100, np.array(ng1).astype(np.float32))
        write("./voices/"+str(q)+"/record_"+str(i+3)+".wav", 44100, np.array(ng2).astype(np.float32))"""

"""for i in range(3):

        ng, fs=  sf.read("./test/record_"+str(i)+".wav", dtype='float32')
        ng1=ng[:int(ng.shape[0]/2)]
        ng2=ng[int(ng.shape[0]/2):]
        write("./test/record_"+str(i)+".wav", 44100, np.array(ng1).astype(np.float32))
        write("./test/record_"+str(i+3)+".wav", 44100, np.array(ng2).astype(np.float32))"""
