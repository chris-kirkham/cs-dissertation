import pretty_midi #http://craffel.github.io/pretty-midi/
import csv
import sys
import itertools
from collections import deque
import numpy
import chordLabelDictionary as labelDictionary
import pprint
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent = 4)

numpy.set_printoptions(suppress = True, threshold = numpy.nan)

inFileMIDI = sys.argv[1]
inFileLabels = sys.argv[2]
rootNote = sys.argv[3]
splitNumBars = int(sys.argv[4]) #0 = leave output as it is, >0 = split output into chunks of corresponding number of bars

#debug mode
debug = True

musicPM = pretty_midi.PrettyMIDI(inFileMIDI)

#turn input into 2D list
#music = list(csv.reader(open(inFileCSV)))
labels = list(csv.reader(open(inFileLabels)))
print(labels)

#INPUT CHECKING:
#check for out-of-order labels
for i, row in enumerate(labels[1:]):
    #print(str(i) + "," + str(len(labels)))
    if(float(labels[i][0]) >= float(row[0])):
        raise RuntimeError("labels timesteps out of order at row " + str(i + 1) + "!")
    
#IMPORTANT: number to scale the time by i.e. the precision of the output. Pretty-midi times things in seconds, and these are rounded to integers to produce the output timesteps.
#so a scalar value of 1 would mean they get rounded to the nearest second, a value of 10 would multiply the times by 10 before rounding, and the times would be rounded to the nearest 100ms;
#a value of 100 would round to the nearest 10ms etc. The higher the scalar, the greater time precision (and so musical detail retained),
#but also the longer sequence length which may make it harder to learn for the network
timeScalar = 20 #precision of 50ms

beats = numpy.multiply(timeScalar, musicPM.get_beats()) #get beats from midi file
if(debug): print(str(len(beats)) + " beats: " + str(beats))

bars = numpy.multiply(timeScalar, musicPM.get_downbeats()) #get downbeats from midi file
#if(debug): print(str(len(bars)) + " downbeats: " + str(bars))

#get notes from midi file
notes = list()
for instrument in musicPM.instruments:
    notes.append(instrument.notes)
    
notes = sorted(list(itertools.chain.from_iterable(notes)), key = lambda x:x.start) #flatten instruments so all notes are in 1D list; sort by start time
for note in notes:
    note.start *= timeScalar
    note.end *= timeScalar

#append end of last note to bars list to allow us to find length of last bar, if necessary
bars = numpy.append(bars, notes[-1].end)

print(str(len(bars)))
#PROCESS LABELS
if(labels[0][0] == '0'): #if labels start at 0
    if(debug): print("labels start at 0")
    for row in labels:
        #get timestep of label if partway through a bar
        if('.' in row[0] ): #if there is a decimal point in bar string
            barNumber = int(float(row[0])) #converting to int drops the decimal part, so just leaves whole bar number
            barTime = bars[barNumber]
            #print(barTime)
            row[0] = round(barTime + ((bars[barNumber + 1] - barTime) * (float(row[0]) % 1))) #time = start of this bar time + (length of this bar * decimal part of bar number)
        else:
            row[0] = bars[int(row[0])] #get timestep of whole bar number by index from bars list
else:
    if(debug): print("labels start at 1")
    for row in labels:
        #get timestep of label if partway through a bar
        if('.' in row[0] ): #if there is a decimal point in bar strings
            barNumber = int(float(row[0])) - 1
            barTime = bars[barNumber]
            #print(barTime)
            row[0] = round(barTime + ((bars[barNumber + 1] - barTime) * (float(row[0]) % 1))) #time = start of this bar time + (length of this bar * decimal part of bar number)
        else:
            row[0] = bars[int(row[0]) - 1]


#labels are now in format [time, label]
#convert [time, label] list to a 1D list of one label per timestep
processedLabels = list()

for i, row in enumerate(labels[1:]):
    timeDiff = (int(round(row[0])) - int(round(labels[i][0])))
    processedLabels.extend(labelDictionary.correctLabel(labels[i][1]) for _ in range(timeDiff))    

#add last label until end of piece
processedLabels.extend(labelDictionary.correctLabel(labels[-1][1]) for _ in range(int(labels[-1][0]), int(notes[-1].end))) 

numpy.savetxt("processedLabelsTest.csv", numpy.asarray(processedLabels), fmt = "%s", delimiter = ",")

#print(processedLabels)   

#PROCESS MUSIC
#round note times to nearest integer
for note in notes:
    note.start = int(round(note.start))
    note.end = int(round(note.end))
        
#process music without bar lines
musicProcessed = numpy.zeros((notes[-1].end, 128))

for note in notes:
    for timestep in range(round(note.start), round(note.end)):
        musicProcessed[timestep][note.pitch] = note.velocity / 128

#split list every splitNumBars bars
splitMusic = list() #list of 2D numpy arrays
splitLabels = list() #1D list of labels

if(splitNumBars > 0):
    numBars = splitNumBars
    startBarTimestep = 0
    endBarTimestep = int(round(bars[min(numBars, len(bars) - 1)]))
    finalBarTimestep = int(round(bars[len(bars) - 1]))
    
    while(endBarTimestep < finalBarTimestep):
        splitMusic.append(musicProcessed[startBarTimestep:endBarTimestep,:])
        splitLabels.append(processedLabels[startBarTimestep:endBarTimestep])    
        
        startBarTimestep = int(round(bars[numBars]))
        numBars += splitNumBars
        endBarTimestep = int(round(bars[min(numBars, len(bars) - 1)]))
        #print(str(numBars) + ", " + str(endBarTimestep))
        
    #append last n bars to split data lists
    finalTimestep = len(musicProcessed) - 1 
    splitMusic.append(musicProcessed[finalBarTimestep:finalTimestep,:])
    splitLabels.append(processedLabels[finalBarTimestep:finalTimestep])   
    
for i in range(len(splitMusic)):
        #if(debug): print(splitMusic[i].shape)            
        #write processed pitches to new file
        with open(inFileMIDI[:-4] + "_pitches_" + rootNote + "_processed_new" + str(i) + ".csv", 'w', newline = '') as f:
            write = csv.writer(f, quoting = csv.QUOTE_ALL)
            write.writerows(splitMusic[i])
                
        #write labels to new file
        with open(inFileMIDI[:-4] + "_labels_" + rootNote + "_processed_new" + str(i) + ".csv", 'w', newline = '') as f:
            write = csv.writer(f, quoting = csv.QUOTE_ALL)
            write.writerow(splitLabels[i])
           
        if(debug): print("length of splitMusic" + str(i) + ": " + str(len(splitMusic[i])) + ", length of splitLabels " + str(i) + "= " + str(len(splitLabels[i]))) 

        
numpy.savetxt("musicProcessedTest.csv", musicProcessed, "%10.5f", delimiter = ",")

if(debug):
    musicProcessedList = musicProcessed.tolist()
    musicProcessedDebug = list()

    for timestep in musicProcessedList:
        rowNotes = list()
        for pitch, velocity in enumerate(timestep): 
            if(float(velocity) > 0): rowNotes.append(pretty_midi.note_number_to_name(pitch))
                
        musicProcessedDebug.append(rowNotes)
        
    numpy.savetxt("musicProcessedNamesTest.csv", numpy.asarray(musicProcessedDebug), fmt = "%s", delimiter = ",")   