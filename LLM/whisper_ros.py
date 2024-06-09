# https://github.com/davabase/whisper_real_time/blob/master/transcribe_demo.py

import argparse
import io
import os
import contextlib
import speech_recognition as sr
import whisper
import torch
import threading
import rclpy
from std_msgs.msg import String

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

import spacy


@contextlib.contextmanager
def temporary_filename(suffix=None):
  """Context that introduces a temporary file.

  Creates a temporary file, yields its name, and upon context exit, deletes it.
  (In contrast, tempfile.NamedTemporaryFile() provides a 'file' object and
  deletes the file as soon as that file object is closed, so the temporary file
  cannot be safely re-opened by another library or process.)

  Args:
    suffix: desired filename extension (e.g. '.mp4').

  Yields:
    The name of the temporary file.
  """
  import tempfile
  try:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_name = f.name
    f.close()
    yield tmp_name
  finally:
    os.unlink(tmp_name)

class whisper_speech():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="medium", help="Model to use",
                            choices=["tiny", "base", "small", "medium", "large"])
        parser.add_argument("--path", default="~/.cache/whisper",
                        help="Where to download the model")  
        parser.add_argument("--non_english", action='store_true',
                            help="Don't use the english model.")
        parser.add_argument("--energy_threshold", default=1000,
                            help="Energy level for mic to detect.", type=int)
        parser.add_argument("--record_timeout", default=7,
                            help="How real time the recording is in seconds.", type=float)
        parser.add_argument("--phrase_timeout", default=2.5,#3
                            help="How much empty space between recordings before we "
                                "consider it a new line in the transcription.", type=float)  

        if 'linux' in platform:
            parser.add_argument("--default_microphone", default='pulse',
                                help="Default microphone name for SpeechRecognition. "
                                    "Run this with 'list' to view available Microphones.", type=str)
        args = parser.parse_args()
        
        # The last time a recording was retreived from the queue.
        self.phrase_time = None
        # Current raw audio bytes.
        self.last_sample = bytes()
        # Thread safe Queue for passing data from the threaded recording callback.
        self.data_queue = Queue()
        # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = args.energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = False
        self.nlp = spacy.load("en_core_web_sm")
        
        # Important for linux users. 
        # Prevents permanent application hang and crash by using the wrong Microphone
        if 'linux' in platform:
            mic_name = args.default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print("Microphone with name "+name+ "found")   
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        self.source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            self.source = sr.Microphone(sample_rate=16000)
            
        # Load / Download model
        model = args.model
        if args.model != "large" and not args.non_english:
            model = model + ".en"
        # LOAD THE MODEL AND SPECIFY THE PATH FOR WHERE TO DOWNLOAD
        self.audio_model = whisper.load_model(model, download_root=args.path)

        self.record_timeout = args.record_timeout
        self.phrase_timeout = args.phrase_timeout

        with temporary_filename() as filename:
            self.temp_file = filename
        #temp_file = NamedTemporaryFile().name
        self.transcription = ['']
        
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        # Cue the user that we're ready to go.
        print("Model loaded.\n")


        self.hallucination_set = {'Thanks for watching!', 'Thanks for watching.', 'Thank you.', 
                                'Thank you!', 'Thank you', 'you', 'You', '', "", ' ', " ", '.',
                                'Good luck.', 'Good luck', 'Good luck!', 'huh', 'Bye', 'Bye.'}
        self.start = False
        # self.stop_listening = self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)
        
        
    def record_callback(self, _, audio):
            """
            Threaded callback function to recieve audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            self.data_queue.put(data)

    def main(self, data):
        
        if data.data =='True':
            self.start = True
            print("recording")
            # Create a background thread that will pass us raw audio bytes.
            # We could do this manually but SpeechRecognizer provides a nice helper.
            self.stop_listening = self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)
            # try: 
        elif data.data=='False':
            print("stopped recording")
            if self.start:
                self.stop_listening(wait_for_stop=False)

            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not self.data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                    self.last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                self.phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    self.last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(self.temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = self.audio_model.transcribe(self.temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # Use spaCy for sentence tokenization
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]

                correct_text = False

                # Publish each sentence as a ROS message
                for sentence in sentences:
                    new_sentence = sentence.strip()
                    
                    if new_sentence.isspace():
                        print("found a space!: {", new_sentence + "}")
                        
                    elif new_sentence not in self.hallucination_set:
                        rospy.sleep(0.1)
                        msg = String()
                        msg.data = new_sentence
                        pub.publish(msg)
                        print(new_sentence)
                        correct_text = True
                    else:
                        print("found it! it's in the list: {", new_sentence, "}")


                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                ## MODIFYING THIS TO EXCLUDE HALLUCINATIONS FROM THE TRANSCRIPT
                if correct_text: 
                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                # os.system('cls' if os.name=='nt' else 'clear')
                # for line in self.transcription:
                #     print(line)
                # # Flush stdout.
                # print('')
                # print("[ Transcript List ]: " + str(self.transcription))

        else:
            rospy.loginfo("whisper Node killed.")
            rospy.signal_shutdown("shutting down whisper node") 

            # except KeyboardInterrupt:
            #     break


        # print("\n\nTranscription:")
        # for line in self.transcription:
        #     print(line)

        # file = open('transcript.txt','w+')
        # for line in self.transcription:
        #     file.write(line+"\n")
        # file.close()

if __name__ == "__main__":

    WS = whisper_speech()
    rclpy.init()
    node = rclpy.create_node("whisper_transcriber")
    pub = node.create_publisher(String, 'speech_text', 10)
    node.create_subscription(String, '/whisper',  WS.main, rclpy.qos.QoSProfile())
    rclpy.spin()
