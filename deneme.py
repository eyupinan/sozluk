import sounddevice as sd

from scipy.io.wavfile import write

sd.default.samplerate = 44100
devices = sd.query_devices()
print(devices)

def callback(indata, frames, time, status):
    write("./anlık_test/stfu.wav", 44100, indata)
    sd.play(indata,device=3, blocking=True)
block_size = int(44100 * 2.5)
_input_stream = sd.InputStream( samplerate=44100,device=1, dtype='float32', callback=callback,channels=2,latency=0,
                                            blocksize=block_size)
_input_stream.start()
"""while _input_stream:
            try:
                frames = _input_stream.read(block_size)[0]
                print(frames)
            except Exception as e:
                print('Error while reading from the audio input: {}'.format(str(e)))
                continue"""

            

