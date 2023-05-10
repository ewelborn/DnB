# https://stackoverflow.com/a/62872679

from pydub import AudioSegment
import math

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + filename
        
        self.audio = AudioSegment.from_file(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        
        # Convert from stereo to mono
        split_audio = split_audio.set_channels(1)
        
        split_audio.export(self.folder + split_filename, format="wav")
        
    def multiple_split(self, sec_per_split):
        total_secs = math.ceil(self.get_duration())
        for i in range(0, total_secs, sec_per_split):
            split_fn = str(i) + '_' + self.filename[:-3] + "wav"
            self.single_split(i, i+sec_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_secs - sec_per_split:
                print('All splited successfully')
                
folder = ''
file = 'Atmosphere Chapter 2 Deeper Drum And Bass 2007.mp3'
split_wav = SplitWavAudioMubin(folder, file)
split_wav.multiple_split(sec_per_split=15)