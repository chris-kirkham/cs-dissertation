import numpy
import matplotlib.pyplot as pyplot
from matplotlib.pyplot import pcolormesh
import csv
import glob
import sys

from argparse import ArgumentParser

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, SimpleRNN, Flatten, TimeDistributed, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import optimizers
from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import process_music

#define command line arguments
argumentParser = ArgumentParser(description = "Given a MIDI file, identify its emotional characteristics from the music itself and generated chord features, visualise this with a graph, and output chord and emotion data to .csv files (\"[MIDI file name]_chords.csv\" and \"[MIDI file name]_emotions.csv)")

argumentParser.add_argument("inputMIDI", metavar = "Input MIDI file", nargs = 1, help = "MIDI file to input")
argumentParser.add_argument("--noOutput", dest = "noOutput", action = "store_true", help = "do not output chord and emotion labels to CSV files (will output by default)")
argumentParser.add_argument("--noGraph", dest = "noGraph", action = "store_true", help = "do not show emotion graph (shown by default)")
argumentParser.add_argument("--debug", dest = "debug", action = "store_true")
argumentParser.set_defaults(noOutput = False, noGraph = False, debug = False)

args = argumentParser.parse_args()
if(args.debug): print(args)

batchSize = 32

with open('../data/chord_identification/label_universe_circle_of_fifths.csv', 'r', newline = '') as f:
    reader = csv.reader(f)
    chordLabelUniverse = next(reader)
    
num_labels = len(chordLabelUniverse) 
max_sequence_timesteps = 250

chordLabelInts = list(range(num_labels))
intToLabelDict = dict(zip(chordLabelInts, chordLabelUniverse)) 

#process input MIDI file
processedMusic = process_music.processMusic(args.inputMIDI[0], max_sequence_timesteps) #process music using process_music's "processMusic" function to process the music
processedMusic3D = processedMusic[numpy.newaxis,:,:]
#load models
chord_identification_model = load_model("../data/saved_models/chord_identification_circle_of_fifths_final.h5")
emotion_recognition_model = load_model("../data/saved_models/emotion_recognition_final.h5")


#predict chords from processed music
chordPredictions = chord_identification_model.predict(processedMusic3D, batch_size = batchSize, verbose = 0)
chordClasses = chordPredictions.argmax(axis = -1)       
chordClassesDebug = [intToLabelDict[chordNum] for chordNum in chordClasses[0]]
if(args.debug): print(chordClassesDebug)

#process predicted chords
processedChords = numpy.zeros(shape = (max_sequence_timesteps, num_labels)) 
for timestep, label in enumerate(chordClasses):
    #change corresponding feature in zeroes array to 1 by getting the label's number from the dictionary
    processedChords[timestep][label] = 1  
    
#predict emotions from processed music and predicted chords
musicAndChords = numpy.hstack((processedMusic, processedChords))
musicAndChords3D = musicAndChords[numpy.newaxis,:,:]
emotionPredictions = emotion_recognition_model.predict(musicAndChords3D, batch_size = batchSize, verbose = 0)

if(args.debug): print(emotionPredictions)

#plot predictions
predictionsX = [p[0] for p in emotionPredictions[0]]
predictionsY = [p[1] for p in emotionPredictions[0]]

#get (closest) emotion labels from predictions 
def getEmotionLabel(valence, arousal):
    if(valence > -0.1 and valence < 0.1):
        if(arousal < 0.25): return "Sleepy"
        if(arousal < 0.75): return "Calm" 
        else: return "Excited"
    if(valence > 0):
        if(arousal < 0.25): return "Peaceful"
        if(arousal < 0.5): return "Relaxed"
        if(arousal < 0.75): return "Pleased"
        else: return "Happy"
    else:
        if(arousal < 0.25): return "Sad"
        if(arousal < 0.5): return "Bored"
        if(arousal < 0.75): return "Nervous"
        else: return "Angry"

        
#valence/arousal line graphs
if not (args.noGraph):
    fig, lines = pyplot.subplots(nrows=1)

    lines.plot(predictionsX)
    lines.plot(predictionsY)
    lines.legend(["Valence", "Arousal"])
    lines.set_xlabel("Timesteps")
    lines.set_ylabel("Values")
    
    pyplot.show()
    
    if(args.debug): print("Graph plotted")

#output chord/emotion predictions to files
if not (args.noOutput):
    midiName = str(args.inputMIDI[0])
    midiName = midiName[midiName.rfind("/") + 1:-4]
    
    with open("chord_predictions/" + midiName + "_chords.csv", 'w', newline = '') as f:
                write = csv.writer(f, quoting = csv.QUOTE_ALL)
                write.writerow(chordClassesDebug)

    if(args.debug): print("Chord predictions saved to chord_predictions/" + midiName + "_chords.csv")
    
    with open("emotion_predictions/" + midiName + "_emotions.csv", 'w', newline = '') as f:
                write = csv.writer(f, quoting = csv.QUOTE_ALL)
                write.writerows(emotionPredictions[0])

    if(args.debug): print("Emotion predictions saved to emotion_predictions/" + midiName + "_emotions.csv")    