import csv
import sys
import glob
import numpy
import itertools
import pprint

#function to convert root numbers (up from C) to letters for file naming
def getRoot(n):
    if (n == 0): return "C"
    if (n == 1): return "C#"
    if (n == 2): return "D"
    if (n == 3): return "D#"
    if (n == 4): return "E"
    if (n == 5): return "F"
    if (n == 6): return "F#"
    if (n == 7): return "G"
    if (n == 8): return "G#"
    if (n == 9): return "A"
    if (n == 10): return "A#"
    if (n == 11): return "B"
    
#function to convert root notes to integers (up from C = 0)
def getNumFromRoot(root):
    if (root == "C"): return 0
    if (root == "C#"): return 1
    if (root == "Db"): return 1
    if (root == "D"): return 2
    if (root == "D#"): return 3
    if (root == "Eb"): return 3
    if (root == "E"): return 4
    if (root == "F"): return 5
    if (root == "F#"): return 6
    if (root == "Gb"): return 6
    if (root == "G"): return 7
    if (root == "G#"): return 8
    if (root == "Ab"): return 9
    if (root == "A"): return 9
    if (root == "A#"): return 10
    if (root == "Bb"): return 10
    if (root == "B"): return 11

csvsDirectory = sys.argv[1]
labelsDirectory = sys.argv[2]
rootNote = sys.argv[3]
labelType = int(sys.argv[4]) #0 = chords (transpose labels), 1 = emotions (do not transpose labels)

rootNoteNumber = getNumFromRoot(rootNote)

pp = pprint.PrettyPrinter(indent = 4)

#debug mode
debug = True

csvs = sorted(glob.glob(csvsDirectory + "/*.csv".lower()))
labels = sorted(glob.glob(labelsDirectory + "/*.csv".lower()))

if(len(csvs) != len(labels)):
    raise RuntimeError("number of csvs and corresponding label files must be the same! Number of csvs: " + str(len(csvs)) + ", number of labels: " + str(len(labels)))
    
#TRANSPOSE EACH MUSIC FILE AND CORRESPONDING LABEL FILE
for i in range(len(csvs)):
    #turn input into 2D list
    data = list(csv.reader(open(csvs[i])))

    #TRANSPOSE CSVS AND WRITE TO NEW FILES
    #remove "_[root note]_processed.csv" from end of inFile    
    outFileCSV = csvs[i][:-4] #remove file extension from filename

    #loop through octaves to transpose and output transposed files (omit 0 (no transposition))
    for n in range(1,12): 
        dataTransposed = list()

        for row in data:
            rowTransposed = numpy.zeros((128))
            
            for pitch, velocity in enumerate(row):
                velocity = float(velocity)
                if(velocity > 0):
                    newPitch = (pitch + n) % 128
                    rowTransposed[newPitch] = velocity
                    
            dataTransposed.append(rowTransposed)
        
        #write file
        with open(''.join((outFileCSV, "_" + getRoot((n + rootNoteNumber) % 12), "_processed.csv")), 'w', newline='') as f:
            write = csv.writer(f, quoting = csv.QUOTE_ALL)
            write.writerows(dataTransposed)
    
    
    #transpose labels and write to new fils
    #read labels into list
    labelData = next(csv.reader(open(labels[i])))

    #remove "_[root note]_labels_processed.csv" from end of inFile    
    outFileLabels = labels[i][:-4]

    for n in range(1,12):
        if(labelType == 0): #chord labels
            labelsTransposed = list()
            #print(labelData)
            for label in labelData:
                if(label != "None-"):
                    #remove root note from label
                    if(label[1] == '#'):
                        root = label[:2]
                        label = label[2:]
                    else:
                        root = label[:1]
                        label = label[1:]
                    
                    if(label[-4:] == "Dim7"):
                        newRootNum = (getNumFromRoot(root) + n) % 12
                        if(newRootNum == 0 or newRootNum == 3 or newRootNum == 6 or newRootNum == 9): #CDim7
                            label = "C-/D#-/F#-/A-Dim7"
                        elif(newRootNum == 1 or newRootNum == 4 or newRootNum == 7 or newRootNum == 10): #C#Dim7
                            label = "C#-/E-/G-/A#-Dim7"
                        else: #DDim7
                            label = "D-/F-/G#-/B-Dim7" 
                    else:
                        label = getRoot((getNumFromRoot(root) + n) % 12) + label
                
                labelsTransposed.append(label)
            
            #write file
            with open(''.join((outFileLabels, "_" + getRoot((n + rootNoteNumber) % 12), "_labels_processed.csv")), 'w', newline='') as f:
                write = csv.writer(f, quoting = csv.QUOTE_ALL)
                write.writerow(labelsTransposed)
        else: #emotion labels
            #write file
            with open(''.join((outFileLabels, "_", str(n), "_labels_processed.csv")), 'w', newline='') as f:
                write = csv.writer(f, quoting = csv.QUOTE_ALL)
                write.writerow(labelData)
            
          